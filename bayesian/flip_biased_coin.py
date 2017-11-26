def likelihood(data, p):
    if data == 'H':
        return p
    else:
        return 1 - p
