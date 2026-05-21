import multiprocessing

def worker(i):
    return i * 2

if __name__ == '__main__':
    with multiprocessing.Pool(4) as p:
        print(p.map(worker, range(4)))
