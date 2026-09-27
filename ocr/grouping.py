

def group_positions(second, th=60):
    seconds = [s for s in second if s is not None ]

    groups = []
    current_group = [seconds[0]]
    for sec in range(1, len(seconds)):
        now = seconds[sec]

        if now -  current_group[-1] < 60:
            current_group.append(now)
        else:
            groups.append(current_group)
            current_group = [now]

    groups.append(current_group)

    return groups