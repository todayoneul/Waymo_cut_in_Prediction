"""과거 1초 궤적과 현재 시점 지도에서 끼어들기 예측용 feature를 계산한다 (미래 정보 없음)."""
import numpy as np

X0, Y0 = 24, 24  # BEV 격자: 열 = x + 24, 행 = 24 - y (1 m/칸, 기준 차량이 원점)
REF_STD_LEN, REF_STD_WID = 4.7, 2.0  # 기준 차량은 고정 표준 크기 사용 (Waymo 차량 길이 5.29m 지름길 차단)


def side_of(seq):
    return np.where(seq[:, -1, 1, 1] >= 0, 1.0, -1.0)


def mirrored(seq):
    # 상대 차량이 항상 기준 차량의 왼쪽(+y)에 오도록 좌우를 뒤집는다.
    s = seq.astype(np.float64).copy()
    sd = side_of(seq)[:, None, None]
    for c in (1, 3, 4):
        s[..., c] *= sd
    return s


def _unwrap_to_last(yaw):
    # ±π 경계에서 2π 튀는 현상을 막되, 현재 시점(마지막 프레임) 각도는 원래 [-π, π] 범위를 유지한다.
    unwrapped = np.unwrap(yaw, axis=1)
    return unwrapped + (yaw[:, -1:] - unwrapped[:, -1:])


def kinematic_features(seq):
    """두 차량의 거리·속도·가속도·방향 변화와 등속 가정 1~3초 뒤 예상 위치."""
    s = mirrored(seq)
    ref, tg = s[:, :, 0], s[:, :, 1]
    f = {}
    rel = tg[:, :, :4] - ref[:, :, :4]
    now = rel[:, -1]
    L = (REF_STD_LEN + tg[:, -1, 5]) / 2
    W = (REF_STD_WID + tg[:, -1, 6]) / 2
    f["fwd"], f["lat"], f["dvx"], f["dvy"] = now[:, 0], now[:, 1], now[:, 2], now[:, 3]  # fwd: 현재 상대 앞뒤 거리 · lat: 현재 좌우 거리 · dvx/dvy: 앞뒤/좌우 상대속도
    f["bumper_gap"], f["rear_gap"], f["edge_gap"] = now[:, 0] - L, -now[:, 0] - L, now[:, 1] - W  # bumper_gap/rear_gap: 차량 길이를 고려한 앞뒤 간격 · edge_gap: 차폭을 고려한 옆 간격
    vr = np.hypot(ref[:, -1, 2], ref[:, -1, 3])
    vt = np.hypot(tg[:, -1, 2], tg[:, -1, 3])
    f["v_ref"], f["v_tgt"] = vr, vt  # v_ref/v_tgt: 기준/상대 차량의 현재 속력
    f["time_gap"], f["speed_ratio"] = now[:, 0] / np.maximum(vr, 0.5), vt / np.maximum(vr, 0.5)  # time_gap: 앞뒤 거리 ÷ 기준 속력 · speed_ratio: 상대/기준 속력 비율
    f["tgt_len"], f["tgt_wid"] = tg[:, -1, 5], tg[:, -1, 6]  # tgt_len/tgt_wid: 상대 차량 길이/너비 (기준 차량 길이는 제외)
    yaw_t, yaw_r = _unwrap_to_last(tg[:, :, 4]), _unwrap_to_last(ref[:, :, 4])
    f["yaw_t"], f["lat_speed_from_heading"] = yaw_t[:, -1], vt * np.sin(yaw_t[:, -1])  # yaw_t: 상대 방향 · lat_speed_from_heading: 속력과 방향으로 계산한 옆 속도
    for k in (2, 5, 10):  # 0.2초, 0.5초, 1초 전과 비교
        dt = k / 10
        f[f"lat_rate_{k}"] = (rel[:, -1, 1] - rel[:, -1 - k, 1]) / dt  # 지난 k프레임 동안 두 차량의 좌우 간격 변화율
        f[f"tgt_y_rate_{k}"] = (tg[:, -1, 1] - tg[:, -1 - k, 1]) / dt  # 지난 k프레임 동안 상대 차량의 좌우 위치 변화율
        f[f"ref_y_rate_{k}"] = (ref[:, -1, 1] - ref[:, -1 - k, 1]) / dt  # 지난 k프레임 동안 기준 차량의 좌우 위치 변화율
        f[f"fwd_rate_{k}"] = (rel[:, -1, 0] - rel[:, -1 - k, 0]) / dt  # 지난 k프레임 동안 두 차량의 앞뒤 간격 변화율
        f[f"tgt_acc_{k}"] = (vt - np.hypot(tg[:, -1 - k, 2], tg[:, -1 - k, 3])) / dt  # 지난 k프레임 동안 상대 차량의 속력 변화율
        f[f"ref_acc_{k}"] = (vr - np.hypot(ref[:, -1 - k, 2], ref[:, -1 - k, 3])) / dt  # 지난 k프레임 동안 기준 차량의 속력 변화율
        f[f"tgt_yaw_rate_{k}"] = np.sin(yaw_t[:, -1] - yaw_t[:, -1 - k]) / dt  # 지난 k프레임 동안 상대 차량 방향 변화의 사인값을 시간으로 나눈 값
        f[f"ref_yaw_rate_{k}"] = np.sin(yaw_r[:, -1] - yaw_r[:, -1 - k]) / dt  # 지난 k프레임 동안 기준 차량 방향 변화의 사인값을 시간으로 나눈 값
        f[f"tgt_vy_change_{k}"] = (tg[:, -1, 3] - tg[:, -1 - k, 3]) / dt  # 지난 k프레임 동안 상대 차량의 좌우 속도 변화율
    f["tgt_vy"], f["tgt_vy_mean"] = tg[:, -1, 3], tg[:, :, 3].mean(1)  # tgt_vy: 현재 좌우 속도 · tgt_vy_mean: 지난 1초 평균 좌우 속도
    f["tgt_vy_min"], f["tgt_vy_max"], f["tgt_vy_std"] = tg[:, :, 3].min(1), tg[:, :, 3].max(1), tg[:, :, 3].std(1)  # tgt_vy_min/max/std: 지난 1초 좌우 속도의 최소/최대/표준편차
    f["yaw_t_mean"], f["yaw_t_min"] = yaw_t.mean(1), yaw_t.min(1)  # yaw_t_mean/min: 지난 1초 상대 차량 방향의 평균/최솟값
    # 11프레임에 2차 곡선을 맞춰 옆 간격과 방향의 추세·휘어짐을 구한다.
    P = np.linalg.pinv(np.vander(np.arange(-10, 1) / 10.0, 3))
    co = rel[:, :, 1] @ P.T
    f["lat_quad"], f["lat_slope_fit"] = co[:, 0], co[:, 1]  # lat_quad: 좌우 간격 2차 추세의 곡률 계수 · lat_slope_fit: 선형 계수
    co = yaw_t @ P.T
    f["yaw_quad"], f["yaw_slope_fit"] = co[:, 0], co[:, 1]  # yaw_quad: 상대 방향 2차 추세의 곡률 계수 · yaw_slope_fit: 선형 계수
    for h in (1, 2, 3):
        f[f"lat_pred_{h}"] = now[:, 1] + now[:, 3] * h  # 현재 좌우 상대속도가 h초 유지된다고 가정한 좌우 간격
        f[f"fwd_pred_{h}"] = now[:, 0] + now[:, 2] * h  # 현재 앞뒤 상대속도가 h초 유지된다고 가정한 앞뒤 간격
        f[f"lat_pred_fit_{h}"] = now[:, 1] + f["lat_slope_fit"] * h + f["lat_quad"] * h * h  # 지난 1초의 좌우 간격 곡선이 h초 이어진다고 가정한 추정 간격
    closing = np.maximum(-now[:, 3], 1e-3)
    f["time_to_edge"] = np.clip(f["edge_gap"] / closing, 0, 20)  # 현재 좌우 접근 속도가 유지될 때 차체 간 옆 간격에 대한 시간 추정치
    f["time_to_center"] = np.clip(now[:, 1] / closing, 0, 20)  # 현재 좌우 접근 속도가 유지될 때 중심 간 옆 간격에 대한 시간 추정치
    return np.column_stack(list(f.values())), list(f)


def _columns(channel, side):
    rows, cols = np.nonzero(channel)
    out = [[] for _ in range(96)]
    for c, yv in zip(cols.tolist(), ((Y0 - rows) * side).tolist()):
        out[c].append(yv)
    return out


def _trace(cols, x0, y0, tol=2.0, max_gap=3):
    # (x0, y0) 근처의 선에서 출발해 앞뒤로 한 칸씩 이어진 점을 따라간다.
    c0, best = int(round(x0 + X0)), None
    for dc in (0, -1, 1, -2, 2):
        c = c0 + dc
        if 0 <= c < 96:
            for yv in cols[c]:
                dist = abs(yv - y0) + 0.2 * abs(dc)
                if abs(yv - y0) <= tol and (best is None or dist < best[0]):
                    best = (dist, c, yv)
    if best is None:
        return None
    _, cs, ys = best
    pts = [(cs - X0, ys)]
    for step in (1, -1):
        cur, slope, gap, c = ys, 0.0, 0, cs + step
        while 0 <= c < 96:
            pred, hit = cur + slope, None
            if cols[c]:
                hit = min(cols[c], key=lambda v: abs(v - pred))
                if abs(hit - pred) > 1.5:
                    hit = None
            if hit is None:
                gap, cur = gap + 1, pred
                if gap > max_gap:
                    break
            else:
                slope = float(np.clip(0.7 * slope + 0.3 * (hit - cur), -1, 1))
                cur, gap = hit, 0
                pts.append((c - X0, hit))
            c += step
    pts.sort()
    return np.array(pts, float) if len(pts) >= 5 else None


def _local(curve, xq, half=8.0):
    # xq 주변 ±8 m 점들에 직선을 맞춰 곡선의 높이와 기울기를 구한다.
    if curve is None:
        return np.nan, np.nan
    xs, ys = curve[:, 0], curve[:, 1]
    if xq < xs[0] - 4 or xq > xs[-1] + 4:
        return np.nan, np.nan
    m = np.abs(xs - xq) <= half
    if m.sum() < 3:
        return np.nan, np.nan
    xm, ym = xs[m], ys[m]
    dx = xm - xm.mean()
    den = (dx * dx).sum()
    slope = (dx * (ym - ym.mean())).sum() / den if den > 0 else 0.0
    return ym.mean() + slope * (xq - xm.mean()), slope


def _one_map_row(ref, tg, sd, bev):
    f = {}
    xt, yt = tg[-1, 0], tg[-1, 1]
    Lt, Wt, Lr = tg[-1, 5], tg[-1, 6], REF_STD_LEN
    lane, line = _columns(bev[0], sd), _columns(bev[1], sd)
    cr, ct = _trace(lane, 0.0, 0.0), _trace(lane, xt, yt)  # 기준 차로, 상대 차로의 중심선
    a0, b0 = _local(cr, xt)[0], _local(ct, xt)[0]
    f["same_lane_trace"] = float(np.isfinite(a0) and np.isfinite(b0) and abs(b0 - a0) < 0.8)  # 두 차량 주변에서 추적한 중심선이 같은 선처럼 보이면 1
    # 차로가 앞뒤로 얼마나 이어지는지 (차로 끝·합류 단서)
    f["ref_lane_ahead"] = cr[-1, 0] if cr is not None else np.nan  # 격자에서 찾은 기준 차로 중심선의 앞쪽 끝 위치
    f["tgt_lane_ahead"] = ct[-1, 0] - xt if ct is not None else np.nan  # 상대 차량 위치에서 추적한 차로 중심선 앞쪽 끝까지의 거리
    f["tgt_lane_behind"] = xt - ct[0, 0] if ct is not None else np.nan  # 상대 차량 위치에서 추적한 차로 중심선 뒤쪽 끝까지의 거리
    # 기준 차량이 자기 차로 안에서 치우친 정도와 그 변화
    ycr0, sr0 = _local(cr, 0.0)
    f["ref_off_now"] = -ycr0  # 기준 차량의 현재 차로 중심선 대비 좌우 편차
    f["ref_lat_vel_lane"] = ref[-1, 3] - ref[-1, 2] * sr0 if np.isfinite(sr0) else np.nan  # 기준 차량의 차로 기울기를 고려한 좌우 속도
    f["ref_off_change_1s"] = (-ycr0) - (ref[0, 1] - _local(cr, ref[0, 0])[0])  # 기준 차량의 차로 중심선 대비 좌우 편차가 1초 동안 변한 양
    # 상대 차량의 위치를 차로 곡선 기준으로 다시 잰다 (굽은 도로 보정)
    d, o = [], []
    for k in (0, 5, 10):
        xq, yq = tg[-1 - k, 0], tg[-1 - k, 1]
        a, sa = _local(cr, xq)
        b, sb = _local(ct, xq)
        d.append(yq - a)
        o.append(yq - b)
        if k == 0:
            f["lane_slope_at_tgt"] = sa  # 상대 차량 앞뒤 위치에서 추정한 기준 차로 중심선의 기울기
            f["tgt_lat_vel_lane"] = tg[-1, 3] - tg[-1, 2] * sa if np.isfinite(sa) else np.nan  # 상대 차량의 기준 차로 기울기 대비 좌우 속도
            f["tgt_lat_vel_ownlane"] = tg[-1, 3] - tg[-1, 2] * sb if np.isfinite(sb) else np.nan  # 상대 차량의 자기 차로 기울기 대비 좌우 속도
            f["tgt_heading_lane"] = tg[-1, 4] - np.arctan(sb) if np.isfinite(sb) else np.nan  # 상대 차량 방향과 자기 차로 방향의 차이
            f["lane_sep"] = b - a  # 상대 차량 앞뒤 위치에서 두 추적 중심선의 좌우 간격
    f["d_ref_lane"], f["d_ref_lane_edge"] = d[0], d[0] - Wt / 2  # d_ref_lane: 상대 중심과 기준 차로 중심의 좌우 거리 · d_ref_lane_edge: 차폭 반영 거리
    f["d_ref_lane_rate_5"], f["d_ref_lane_rate_10"] = (d[0] - d[1]) / 0.5, (d[0] - d[2]) / 1.0  # d_ref_lane_rate_5/10: 지난 0.5초/1초의 기준 차로 중심 대비 거리 변화율
    f["own_off"] = o[0]  # 상대 차량 중심의 자기 차로 중심선 대비 좌우 편차
    f["own_off_rate_5"], f["own_off_rate_10"] = (o[0] - o[1]) / 0.5, (o[0] - o[2]) / 1.0  # own_off_rate_5/10: 지난 0.5초/1초의 자기 차로 중심 대비 편차 변화율
    ok = np.isfinite(f["d_ref_lane_rate_10"]) and np.isfinite(d[0])
    f["time_to_ref_lane"] = (np.clip((d[0] - Wt / 2 - 1.75) / max(-f["d_ref_lane_rate_10"], 1e-2), 0, 20)  # 현재 접근 추세가 이어진다고 가정한 기준 차로 접근 시간 관련 추정치
                             if ok else np.nan)
    for h in (20, 40):  # 앞쪽에서 두 차로가 가까워지면 합류 구간
        f[f"lane_sep_ahead_{h}"] = _local(ct, xt + h)[0] - _local(cr, xt + h)[0]  # 상대 차량 앞 h미터에서 두 추적 차로 중심선 사이의 간격
    # 상대 차량 옆의 차선까지 거리
    c = int(round(xt + X0))
    near = [v for cc in (c - 1, c, c + 1) if 0 <= cc < 96 for v in line[cc]]
    inner = [v for v in near if -1.0 < v < yt + 0.3]
    outer = [v for v in near if v > yt + 0.3]
    f["inner_line_dist"] = (yt - Wt / 2) - max(inner) if inner else np.nan  # 상대 차량 안쪽 차체와 인근 안쪽 도로 경계의 근사 거리
    f["outer_line_dist"] = min(outer) - (yt + Wt / 2) if outer else np.nan  # 상대 차량 바깥쪽 차체와 인근 바깥쪽 도로 경계의 근사 거리
    f["lines_between"] = float(len({round(v) for v in near if 0.8 < v < yt - 0.8}))  # 두 차량 사이에서 격자로 감지된 도로 경계 좌우 위치의 개수
    # 주변 차량: 기준 차로 앞 공간, 끼어들 자리, 상대 차량 앞차까지 거리
    rr, cc_ = np.nonzero(bev[4])
    if len(rr):
        ox, oy = cc_ - X0, (Y0 - rr) * sd
        dr = oy - np.interp(ox, cr[:, 0], cr[:, 1]) if cr is not None else oy.astype(float)
        dt_ = oy - np.interp(ox, ct[:, 0], ct[:, 1]) if ct is not None else oy - yt
        inr, intg = np.abs(dr) < 1.3, np.abs(dt_) < 1.3
        a = ox[inr & (ox > Lr / 2 + 0.5)]
        f["ref_lead_dist"] = a.min() - Lr / 2 if len(a) else 80.0  # 기준 차로의 기준 차량 앞쪽 차량까지의 근사 거리, 없으면 80m
        a = ox[inr & (ox < -Lr / 2 - 0.5)]
        f["ref_follow_dist"] = -a.max() - Lr / 2 if len(a) else 30.0  # 기준 차로의 기준 차량 뒤쪽 차량까지의 근사 거리, 없으면 30m
        a = ox[inr]
        f["slot_dist"] = np.abs(a - xt).min() if len(a) else 80.0  # 상대 차량의 앞뒤 위치에서 기준 차로 주변 차량까지의 최소 앞뒤 거리, 없으면 80m
        a = ox[intg & (ox > xt + Lt / 2 + 0.5)]
        f["tgt_lead_dist"] = a.min() - xt - Lt / 2 if len(a) else 80.0  # 상대 차로에서 상대 차량 앞쪽 차량까지의 근사 거리, 없으면 80m
        a = ox[intg & (ox < xt - Lt / 2 - 0.5)]
        f["tgt_follow_dist"] = xt - Lt / 2 - a.max() if len(a) else 60.0  # 상대 차로에서 상대 차량 뒤쪽 차량까지의 근사 거리, 없으면 60m
        f["n_other_near"] = float((np.hypot(ox - xt, oy - yt) < 20).sum()) / 10  # 상대 차량 20m 이내 주변 차량 점유 격자 수 ÷ 10 (실제 차량 대수 아님)
        f["n_other_ref_lane_ahead"] = float((inr & (ox > 0)).sum()) / 10  # 기준 차로 앞쪽 주변 차량 점유 격자 수 ÷ 10 (실제 차량 대수 아님)
    else:
        f.update(ref_lead_dist=80.0, ref_follow_dist=30.0, slot_dist=80.0, tgt_lead_dist=80.0,
                 tgt_follow_dist=60.0, n_other_near=0.0, n_other_ref_lane_ahead=0.0)
    f["ref_lead_minus_tgt"] = f["ref_lead_dist"] - (xt - Lt / 2 - Lr / 2)  # 기준 차량 앞 공간과 상대 차량 앞뒤 위치를 비교한 근사값
    vt, vr = np.hypot(tg[-1, 2], tg[-1, 3]), np.hypot(ref[-1, 2], ref[-1, 3])
    f["tgt_lead_headway"] = min(f["tgt_lead_dist"] / max(vt, 0.5), 20)  # 상대 차량 앞 공간 ÷ 상대 속력, 최대 20초
    f["ref_lead_headway"] = min(f["ref_lead_dist"] / max(vr, 0.5), 20)  # 기준 차량 앞 공간 ÷ 기준 속력, 최대 20초
    return f


def map_features(seq, bev):
    """현재 시점 BEV(차로 중심선, 차선, 주변 차량)에서 차로 기준 위치와 주변 공간을 계산한다."""
    s, side = mirrored(seq), side_of(seq)
    rows = [_one_map_row(s[i, :, 0], s[i, :, 1], side[i], bev[i]) for i in range(len(s))]
    names = list(rows[0])
    return np.array([[r[n] for n in names] for r in rows], float), names


# ---------- 지도 feature를 묶음 단위로 계산 ----------

def build_map_features(seq, packed_bev, batch_size=512):
    """BEV를 한 묶음씩 풀어 지도 숫자 변수로 변환한다."""
    batches = []
    for start in range(0, len(seq), batch_size):
        batch = slice(start, start + batch_size)
        bev = np.unpackbits(packed_bev[batch], axis=-1, count=96)
        values, names = map_features(seq[batch], bev)
        batches.append(values)
    return np.vstack(batches), names
