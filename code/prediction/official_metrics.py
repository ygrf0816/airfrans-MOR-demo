"""B 链路官方口径力系数评估 —— 判定实验（judge_lifting.py）专用工具。

修复记录（2026-10-03，用户 4 点指示第 4 条）：
  - 旧版 `force_coefficients_from_cp` 的 C_L 公式（256 点机械对半分 + 弦向 np.trapz
    投影 + C_D_pressure 硬编码 0.0）已删除：对称翼型 AoA=0 时会给出非零 C_L
    （真实值为 0），这是错的公式，不是"简化"，不能进任何报告。
  - 官方力系数口径 = `metrics.Compute_coefficients`（01_initial/AirfRANS-main/
    metrics.py:186-234）：逐点 U 梯度 -> 壁面剪应力 + 压力项 x 真实法向 x 真实
    cell 弧长 -> 按 AoA 旋转 -> coef = 2*force_rot/U∞²。论文
    _pdf_full.txt:262-266/375-377 也是这套口径（"computed as a post-processing
    from the unnormalized regressed fields"）。
  - 本模块只保留：官方真值 C_L/C_D 的计算封装（委托 pyvista，一次性）+ 压力项
    版 C_L 的判定用近似公式（真实法向+真实弧长，缺摩擦项），供 judge_lifting.py
    做"相对差 <5% 就采用/否则删除"的分叉判定。

口径铁律：只有判定实验输出的 `judge_lifting_full.json` 里 `branch` 字段决定
train_B_transformer.py 最终报哪套力系数字；C_D 在任何分叉下都不报（官方 C_D
摩擦主导，Cp 单口径恢复不了摩擦项，硬报一个 0 会误导）。
"""
import os
import numpy as np


def _decode(name):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geometry_input import decode_sim
    return decode_sim(name)


def official_force_split(intern, airfoil, Uinf, AoA_deg):
    """官方口径力系数**分解版**（01_initial/AirRANS-main/metrics.py:186-234 的自包含实现）：

    返回 (full, pressure_only, friction_only)，每个都是 [C_D, C_L]：
      full          = 2·rot(α)(Wss_int − WP_int)/U∞²   —— 官方 Table 5 报的就是这个
      pressure_only = 2·rot(α)(0 − WP_int)/U∞²          —— 只还原壁面 Cp 时能算到的部分
      friction_only = 2·rot(α)(Wss_int)/U∞²             —— 需要逐点速度梯度才能算的部分
    用途：给"B 链路只还原壁面 Cp，因此能不能报 C_D"提供可存档的量化依据。
    口径见 judge_friction_split：摩擦项占 C_L 约 0.02%、占 C_D 约 58%。

    实现逐字对应官方 Wss 公式、reorganize、force 积分，不依赖 airfrans 库：
      1. intern.compute_derivative('U') -> pred_grad（逐点速度梯度）
      2. 把 intern 的梯度/压力按坐标精确匹配（官方 reorganize 口径）重排到 airfoil 表面
      3. Wss 逐 airfoil 表面点算（官方 metrics.py::WallShearStress，用该点法向）
      4. Wss_int / WP_int 用真实 cell 弧长 Length 积分，再按 AoA 旋转
    """
    import numpy as np
    intern = intern.copy()
    airfoil = airfoil.copy()

    intern = intern.compute_derivative(scalars='U', gradient='pred_grad')
    in_pts = intern.points[:, :2]
    out_pts = airfoil.points[:, :2]
    NU = 1.56e-5

    def _reorg(qty):
        """官方 reorganize.py 逐字等价：按 out_pts 每行在 in_pts 里找精确坐标匹配的索引。

        官方在末尾 `assert (in_order_points[idx] == out_order_points).all()`；
        匹配失败时必须报错，不能退回 -1（会静默取到最后一个点的值，给出错数）。
        """
        idx = np.zeros(out_pts.shape[0])
        for i in range(out_pts.shape[0]):
            w = np.argwhere((in_pts == out_pts[i]).all(axis=1))
            if not len(w):
                raise RuntimeError(
                    "reorganize: 内部网格里找不到与翼型点精确匹配的坐标 i=%d" % i)
            idx[i] = w[0, 0]
        idx = idx.astype(int)
        assert (in_pts[idx] == out_pts).all()
        return qty[idx]

    grad_intern = intern.point_data['pred_grad'].reshape(-1, 3, 3)[:, :2, :2]
    p_intern = intern.point_data['p']
    normals_a = airfoil.point_data['Normals'][:, :2]

    grad_surf = _reorg(grad_intern)
    p_surf = _reorg(p_intern)
    S = 0.5 * (grad_surf + grad_surf.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    shear = 2 * NU * S
    # 官方 metrics.py:209 传的是 -aerofoil.point_data['Normals']（外法向取负），
    # 漏掉这个负号会把摩擦项整体反号：C_L 只差 0.04%（影响小），
    # 但 C_D 会从 +0.0122 变成 -0.0020（摩擦占 C_D 的 58%），"官方真值"直接作废。
    Wss = (shear * (-normals_a).reshape(-1, 1, 2)).sum(axis=2)

    airfoil.point_data['wallShearStress'] = Wss
    airfoil.point_data['p'] = p_surf
    airfoil = airfoil.ptc(pass_point_data=True)

    WP_int = -airfoil.cell_data['p'][:, None] * airfoil.cell_data['Normals'][:, :2]
    WP_int = (WP_int * airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)
    Wss_int = (airfoil.cell_data['wallShearStress']
               * airfoil.cell_data['Length'].reshape(-1, 1)).sum(axis=0)

    alpha = AoA_deg * np.pi / 180
    basis = np.array([[np.cos(alpha), np.sin(alpha)],
                      [-np.sin(alpha), np.cos(alpha)]])

    def _rot(v):
        return 2 * (basis @ v) / Uinf ** 2

    full = _rot(Wss_int - WP_int)
    pressure_only = _rot(0.0 - WP_int)
    friction_only = _rot(Wss_int)
    return (np.array([full[0], full[1]], dtype=np.float64),
            np.array([pressure_only[0], pressure_only[1]], dtype=np.float64),
            np.array([friction_only[0], friction_only[1]], dtype=np.float64))


def official_force_coefficients(intern, airfoil, Uinf, AoA_deg):
    """兼容入口：只返回官方全量 [C_D, C_L]。分解版见 official_force_split。"""
    return official_force_split(intern, airfoil, Uinf, AoA_deg)[0]


def pressure_term_lift_cp(cp_curve, normals, dl, name, n_le_first_half=None):
    """压力项版 C_L（判定用近似，缺摩擦项）：
      F_p = Σ (p·n_in)·ds  （n_in 是 .vtp 存的内法向，和官方 Compute_coefficients
      里 -p·n 的 n 是外法向，方向一致，只是命名不同）
      旋转到来流系 -> 2·F_rot[1]/U∞² 取 y 分量 = 升力（未含摩擦，缺项）。
    cp_curve: (256,) 弧长归一化 Cp（已乘 q=0.5U∞² 还原回 p/q）
    normals:  (994,2) .vtp 逐点法向（内法向），dl: (994,) 逐段弧长
    需要 .vtp 原始 994 点法向+弧长，不是 256 点采样曲线，故签名分开传。
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from geometry_input import decode_sim
    d = decode_sim(name)
    Uinf = d["U_inf"]; AoA = d["AoA_deg"] * np.pi / 180.0
    q = 0.5 * Uinf ** 2
    p = cp_curve * q  # Cp -> p（逐 256 采样点）
    # 把 256 点采样值线性插值回 994 点轮廓（法向/弧长在 994 点尺度）
    import numpy as np
    s256 = np.linspace(0, 1, len(cp_curve))
    s994 = np.cumsum(dl) / dl.sum()
    p994 = np.interp(s994, s256, p)
    F = (p994[:, None] * normals) * dl[:, None]
    F = F.sum(axis=0)
    basis = np.array([[np.cos(AoA), np.sin(AoA)], [-np.sin(AoA), np.cos(AoA)]])
    Frot = basis @ F
    C_L_pressure = 2 * Frot[1] / (Uinf ** 2)
    return C_L_pressure


def anchored_surface(name, task_root):
    """翼型表面在**前缘锚定序**下的 (normals (N,2), dl (N,), s (N,))。

    锚定规则与 workstation_pod.wall_cp(anchor_le=True) 逐行一致：
      1) 闭环起点取 x 最小点（前缘）；
      2) 若走到一半时 y 在下降（说明先走了下表面）则整条翻转，
         保证前半 = 上表面、s = 0 在前缘。
    只有这样，256 点 Cp 曲线按 s 插值回 994 点时才和几何对得上；
    用 _cycle_from_edges 的任意起点会让压力分布相对几何旋转（差 ~14% 周长），
    积分出来的力是错的。
    """
    import os
    import numpy as np
    import vtkxml2 as vx
    from wall_force import _cycle_from_edges, _flat_int_da
    import xml.etree.ElementTree as ET

    fp = os.path.join(task_root, name, name + "_aerofoil.vtp")
    pts, pdata, _, _, _ = vx.read_vtkxml(fp)
    root = ET.parse(fp).getroot()
    piece = next(e for e in root.iter() if e.tag == "Piece")
    conn = off = None
    for da in piece.find("Lines").findall("DataArray"):
        if da.get("Name") == "connectivity":
            conn = _flat_int_da(da)
        elif da.get("Name") == "offsets":
            off = _flat_int_da(da)
    loop = _cycle_from_edges(conn, off, pts[:, :2])
    N = len(loop)
    X = np.vstack([pts[loop, :2], pts[loop[0], :2]])       # N+1 点（闭环）
    n_in = pdata["Normals"][loop][:, :2]                   # N 个法向
    le = int(np.argmin(X[:, 0]))
    Xr = np.roll(X, -le, axis=0)
    Nr = len(Xr)
    reverse = Xr[Nr // 2, 1] < Xr[Nr // 2 - 1, 1]
    if reverse:
        Xr = Xr[::-1]
    nr = np.roll(n_in, -le, axis=0)
    if reverse:
        nr = nr[::-1]
    Xr_N = Xr[:N]
    d = np.linalg.norm(np.diff(np.vstack([Xr_N, Xr_N[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])[:-1]
    s = s / s[-1]
    return nr, d, s


def cp_to_pressure_CL(cp_curve, name, task_root):
    """一条 Cp 曲线（256 点、前缘锚定、s=linspace(0,1)）-> 压力项 C_L。

    真实法向 + 真实弧长，缺壁面剪应力项。实测缺项对 C_L 的影响 = 0.02%
    （摩擦项贡献 C_L 的 0.0002，贡献 C_D 的 58%），故 C_L 可用、C_D 不可用。
    """
    import numpy as np
    d = _decode(name)
    Uinf = d["U_inf"]; AoA = d["AoA_deg"] * np.pi / 180.0
    n_in, dl, s = anchored_surface(name, task_root)
    p994 = np.interp(s, np.linspace(0, 1, len(cp_curve)), np.asarray(cp_curve)) * (0.5 * Uinf ** 2)
    F = ((p994[:, None] * n_in) * dl[:, None]).sum(axis=0)
    basis = np.array([[np.cos(AoA), np.sin(AoA)], [-np.sin(AoA), np.cos(AoA)]])
    return float(2 * (basis @ F)[1] / Uinf ** 2)
