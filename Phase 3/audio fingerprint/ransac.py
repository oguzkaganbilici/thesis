import numpy as np

def ransac(matches, n_iter = 3000, tolerance=20):
    matches = np.array(matches) # x = t_hl, y = t_full
    x = matches[:, 0] # tüm highlights zamanları
    y = matches[:, 1] # tüm full zamanları

    best_count = 0
    best_model = None

    best_inlier_mask = np.zeros(len(matches), dtype=bool)

    for _ in range(n_iter):
        idx = np.random.choice(len(matches), 2, replace=False)
        dot1 = matches[idx[0]]
        dot2 = matches[idx[1]]

        if (dot2[0] - dot1[0]) == 0: # .. / 0 olmaması icin
            continue
    
        slope = (dot2[1] - dot1[1]) / (dot2[0] - dot1[0])

        if not(0.9 < slope < 1.1): # egim yakın degilse adayı ele
            continue

        intersection = dot1[1] - slope * dot1[0]

            
        y_pred = slope * x + intersection # her nokta icin prediction

        distance = np.abs(y - y_pred) # her noktaya olan uzaklıgı

        inlier_mask = distance < tolerance # belirli bir toleranstan yakın olanlar

        inlier_count = np.sum(inlier_mask) # kac tane yakın deger var ise

        if inlier_count > best_count:
            best_count = inlier_count
            best_inlier_mask = inlier_mask 
            best_model = (slope, intersection)


    return best_model, best_inlier_mask


def sequential_ransac(matches,max_turns=30, MIN_INLIER = 50, MAX_X_ARALIGI=5000):
    remain = np.array(matches)
    segments = []
    
    for _ in range(max_turns):
        model, inlier_mask = ransac(remain)
        inlier_count = np.sum(inlier_mask) 

        if inlier_count < MIN_INLIER:
            break

        segment = remain[inlier_mask]
        x_araligi = segment[:, 0].max() - segment[:, 0].min()
        remain = remain[~inlier_mask]


        if x_araligi < MAX_X_ARALIGI:
            segments.append(segment)

    return segments


