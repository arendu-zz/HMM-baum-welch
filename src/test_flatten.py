__author__ = 'arenduchintala'


def flatten_backpointers(bt):
    reverse_bt = []
    while len(bt) > 0:
        x = bt.pop()
        reverse_bt.append(x)
        if len(bt) > 0:
            bt = bt.pop()
    reverse_bt.reverse()
    return reverse_bt


a = [1]
b = [a,5]
c = [b,6]

print 'out',flatten_backpointers(c)