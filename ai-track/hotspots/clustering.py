import numpy as np
from sklearn.cluster import DBSCAN

METERS_PER_DEGREE = 111000


def find_hotspots(reports, eps_meters=100, min_samples=3):
    if len(reports) == 0:
        return []

    eps_degrees = eps_meters / METERS_PER_DEGREE

    coordinates = np.array([[r["lat"], r["lng"]] for r in reports])

    clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_

    hotspots = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue

        member_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        member_reports = [reports[i] for i in member_indices]

        center_lat = float(np.mean([r["lat"] for r in member_reports]))
        center_lng = float(np.mean([r["lng"] for r in member_reports]))

        hotspots.append(
            {
                "cluster_id": int(cluster_id),
                "report_ids": [r["id"] for r in member_reports],
                "center_lat": center_lat,
                "center_lng": center_lng,
                "report_count": len(member_reports),
            }
        )

    return hotspots


if __name__ == "__main__":
    reports = [
        # نقطة ساخنة 1: 5 بلاغات قريبين من بعض حوالين (30.0444, 31.2357)
        {"id": "r1", "lat": 30.0444, "lng": 31.2357},
        {"id": "r2", "lat": 30.0446, "lng": 31.2358},
        {"id": "r3", "lat": 30.0443, "lng": 31.2359},
        {"id": "r4", "lat": 30.0447, "lng": 31.2355},
        {"id": "r5", "lat": 30.0442, "lng": 31.2356},
        # نقطة ساخنة 2: 4 بلاغات قريبين من بعض حوالين (30.0500, 31.2500)
        {"id": "r6", "lat": 30.0500, "lng": 31.2500},
        {"id": "r7", "lat": 30.0502, "lng": 31.2501},
        {"id": "r8", "lat": 30.0498, "lng": 31.2502},
        {"id": "r9", "lat": 30.0501, "lng": 31.2498},
        # 3 بلاغات معزولة، بعيدة عن الكل وعن بعض
        {"id": "r10", "lat": 29.9000, "lng": 31.1000},
        {"id": "r11", "lat": 30.2000, "lng": 31.5000},
        {"id": "r12", "lat": 29.7000, "lng": 30.9000},
    ]

    hotspots = find_hotspots(reports)

    print(f"عدد النقاط الساخنة المكتشفة: {len(hotspots)}")
    print()

    all_hotspot_ids = set()
    for hotspot in hotspots:
        print(f"نقطة ساخنة cluster_id={hotspot['cluster_id']}:")
        print(f"  report_ids: {hotspot['report_ids']}")
        print(f"  center_lat: {hotspot['center_lat']}")
        print(f"  center_lng: {hotspot['center_lng']}")
        print(f"  report_count: {hotspot['report_count']}")
        print()
        all_hotspot_ids.update(hotspot["report_ids"])

    isolated_ids = {"r10", "r11", "r12"}
    overlap = isolated_ids & all_hotspot_ids
    if overlap:
        print(f"تحذير: البلاغات المعزولة دي ظهرت غلط في نقطة ساخنة: {overlap}")
    else:
        print(f"تأكيد: البلاغات المعزولة {sorted(isolated_ids)} لم تظهر في أي نقطة ساخنة (صحيح).")
