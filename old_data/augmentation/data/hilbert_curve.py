# 一阶编码按走向顺序编区号
import numpy as np
import matplotlib.pyplot as plt

hilbert_map = {
    'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
    'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
    'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
    'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
}

def point_to_hilbert(x, y, order):
    current_square = 'a'
    position = 0
    for i in range(order - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position

# print(point_to_hilbert(6,2,6))
'''
points = [(x, y) for x in range(64) for y in range(64)]
sorted_points = sorted(points, key=lambda k: point_to_hilbert(k[0], k[1], 16))
# print('\n'.join('%s,%s' % x for x in sorted_points))
sorted_points = np.array(sorted_points)
plt.figure()
plt.plot(sorted_points[:, 0], sorted_points[:, 1])
plt.show()
'''

un_hilbert_map = {
    'a': { 0: (0, 0,'d'), 1: (0, 1,'a'), 3: (1, 0,'b'),  2: (1, 1,'a')},
    'b': { 2: (0, 0,'b'), 1: (0, 1,'b'), 3: (1, 0,'a'),  0: (1, 1,'c')},
    'c': { 2: (0, 0,'c'), 3: (0, 1,'d'), 1: (1, 0,'c'),  0: (1, 1,'b')},
    'd': { 0: (0, 0,'a'), 3: (0, 1,'c'), 1: (1, 0,'d'),  2: (1, 1,'d')}
}

def hilbert_to_point(d , order):
    current_square = 'a'
    x = y = 0
    for i in range(order - 1, -1, -1):
        # 3的二进制为11，然后左移2i倍，与d取按位与后右移2i倍，得到象限编码
        mask = 3 << (2*i)
        quad_position = (d & mask) >> (2*i)
        quad_x, quad_y, current_square=un_hilbert_map[current_square][quad_position]
        # print(quad_x,quad_y)
        # 不断累加x，y的值，最后总得到解码结果
        x |= 1 << i if quad_x else 0
        y |= 1 << i if quad_y else 0
    return x,y

# print(hilbert_to_point(3000,16))


