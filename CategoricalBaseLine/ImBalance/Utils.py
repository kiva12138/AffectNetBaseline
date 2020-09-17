import numpy as np

class MeanManager(object):
    def __init__(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

class AccuracyManager(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_correct  = np.array(list(0. for i in range(self.num_classes)))
        self.class_total    = np.array(list(0. for i in range(self.num_classes)))
        self.class_accuracy = np.array(list(0. for i in range(self.num_classes)))
        self.ma = 0.0
        self.accuracy = 0.0

    def reset(self):
        self.class_correct  = np.array(list(0. for i in range(self.num_classes)))
        self.class_total    = np.array(list(0. for i in range(self.num_classes)))
        self.class_accuracy = np.array(list(0. for i in range(self.num_classes)))
        self.ma = 0.0
        self.accuracy = 0.0

    def update(self, cls, right):
        if right:
            self.class_correct[cls] += 1
        self.class_total[cls] += 1
        self.class_accuracy[cls] = self.class_correct[cls] / self.class_total[cls]
        self.ma = sum(self.class_accuracy) / self.num_classes
        self.accuracy = sum(self.class_correct) / sum(self.class_total)

if __name__ == "__main__":

    print("Testing AccuracyManager")

    acc = AccuracyManager(num_classes=8)

    acc.update(0, True)
    acc.update(0, True)
    acc.update(0, True)
    acc.update(1, True)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(2, True)
    acc.update(2, False)
    print(acc.accuracy)
    print(acc.ma)
    print(acc.class_correct)
    print(acc.class_total)
    print(acc.class_accuracy)

    acc.reset()
    acc.update(0, True)
    acc.update(0, True)
    acc.update(0, True)
    acc.update(1, True)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(1, False)
    acc.update(2, True)
    acc.update(2, False)
    print(acc.accuracy)
    print(acc.ma)
    print(acc.class_correct)
    print(acc.class_total)
    print(acc.class_accuracy)
