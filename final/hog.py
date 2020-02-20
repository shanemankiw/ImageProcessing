import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class Hog_descriptor():

    def __init__(self, img, cell_size=11, bin_size=8):
        self.img = img
        
        '''转换为灰度图'''
        if len(img.shape) > 2:
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        self.img = np.sqrt(img / float(np.max(img)))
        self.img = self.img * 255
        '''单元格的大小'''
        self.cell_size = cell_size
        '''将0-360度分为几个区间'''
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        #assert type(self.angle_unit) == int, "bin_size should be divisible by 360"


    def extract(self):
        '''
        提取特征的主函数
        '''
        height, width = self.img.shape
        
        # 为了尺度问题考虑，设置了一个备份
        #copy_img = np.zeros([math.ceil(height / self.cell_size)*self.cell_size, \
        #math.ceil(width / self.cell_size)*self.cell_size])
        #copy_img[:height, :width] = self.img
        
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((math.ceil(height / self.cell_size), math.ceil(width / self.cell_size), self.bin_size))
        
        '''计算每个单元格的梯度'''
        '''边界效应真实麻烦'''
        for i in range(cell_gradient_vector.shape[0]-1):
            for j in range(cell_gradient_vector.shape[1]-1):
                
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)
        '''right border'''
        for i in range(cell_gradient_vector.shape[0]-1):
            cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 -self.cell_size:]
            cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             -self.cell_size:]
            cell_gradient_vector[i][cell_gradient_vector.shape[1]-1] = self.cell_gradient(cell_magnitude, cell_angle)
        
        '''down border'''
        for j in range(cell_gradient_vector.shape[1]-1):
            cell_magnitude = gradient_magnitude[-self.cell_size:,
                                 j * self.cell_size:(j + 1) * self.cell_size]
            cell_angle = gradient_angle[-self.cell_size:,
                             j * self.cell_size:(j + 1) * self.cell_size]
            cell_gradient_vector[cell_gradient_vector.shape[0]-1][j] = self.cell_gradient(cell_magnitude, cell_angle)
        '''last corner'''
        cell_magnitude = gradient_magnitude[-self.cell_size:,
                                 -self.cell_size:]
        cell_angle = gradient_angle[-self.cell_size:,
                             -self.cell_size:]
        cell_gradient_vector[cell_gradient_vector.shape[0]-1][cell_gradient_vector.shape[1]-1] = self.cell_gradient(cell_magnitude, cell_angle)

        '''将直方图结果输出到图片'''
        #hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        
        
        #hog_vector = []
        #'''获取hog特征向量'''
        '''normalization,用了L2范数'''
        '''和论文稍微有一点不同，overlap了'''
        # 创建一个bool数组，记录是否已经归一化
        done_before = np.zeros([cell_gradient_vector.shape[0], cell_gradient_vector.shape[1]],bool)
        for i in range(cell_gradient_vector.shape[0]-1):
            for j in range(cell_gradient_vector.shape[1]-1):
                #每个块有4个单元格
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                
                # 计算L2范数
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                
                if magnitude != 0:
                    #normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    #block_vector = normalize(block_vector, magnitude)
                    if not done_before[i][j]:
                        cell_gradient_vector[i][j] = cell_gradient_vector[i][j] / magnitude
                        done_before[i][j] = True
                    if not done_before[i][j+1]:
                        cell_gradient_vector[i][j+1] = cell_gradient_vector[i][j+1] / magnitude
                        done_before[i][j+1] = True
                    if not done_before[i+1][j]:
                        cell_gradient_vector[i+1][j] = cell_gradient_vector[i+1][j] / magnitude
                        done_before[i+1][j] = True
                    if not done_before[i+1][j+1]:
                        cell_gradient_vector[i+1][j+1] = cell_gradient_vector[i+1][j+1] / magnitude
                        done_before[i+1][j+1] = True
                #hog_vector.append(block_vector)
        
        # 最后一步
        hog_pixel = self.get_pixel_max(cell_gradient_vector)

        return hog_pixel

    def global_gradient(self):
        '''
        获取全局的梯度
        包括梯度值和梯度方向
        '''
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        '''
        根据幅度和angle
        获取我们最终需要的那个特征向量
        '''
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))

        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        '''
        决定正切角的区间
        '''
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        '''
        梯度方向可视化
        '''
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image

    def get_pixel_max(self, cell_gradient):
        '''
        根据梯度绘制像素级别histogram
        '''
        image = np.zeros_like(self.img)
        #cell_width = self.cell_size / 2
        #max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]-1):
            for y in range(cell_gradient.shape[1]-1):
                cell_grad = cell_gradient[x][y]
                # 根据论文中的东西调整
                max_hog = cell_grad.max()
                for height in range(x * self.cell_size, (x+1) * self.cell_size):
                    for width in range(y * self.cell_size, (y+1) * self.cell_size):
                        image[height][width] = max_hog
        
        for x in range(cell_gradient.shape[0]-1):
                cell_grad = cell_gradient[x][cell_gradient.shape[1]-1]
                # 根据论文中的东西调整
                max_hog = cell_grad.max()
                for height in range(x * self.cell_size, (x+1) * self.cell_size):
                    for width in range(y * self.cell_size, self.img.shape[1]):
                        image[height][width] = max_hog
        
        for y in range(cell_gradient.shape[1]-1):
                cell_grad = cell_gradient[cell_gradient.shape[0]-1][y]
                # 根据论文中的东西调整
                max_hog = cell_grad.max()
                for height in range(x * self.cell_size, self.img.shape[0]):
                    for width in range(y * self.cell_size, (y+1) * self.cell_size):
                        image[height][width] = max_hog
        
        cell_grad = cell_gradient[cell_gradient.shape[0]-1][cell_gradient.shape[1]-1]
        # 根据论文中的东西调整
        max_hog = cell_grad.max()
        for height in range(self.img.shape[0] - self.img.shape[0]%self.cell_size, self.img.shape[0]):
                for width in range(self.img.shape[1] - self.img.shape[1]%self.cell_size, self.img.shape[1]):
                    image[height][width] = max_hog
        
        return image

'''Testing'''
if __name__ == '__main__':
    img = cv2.imread('imgs/t001.png', cv2.IMREAD_GRAYSCALE)
    hog = Hog_descriptor(img, cell_size=11, bin_size=8)
    histogram = hog.extract()
    #plt.imshow(image, cmap=plt.cm.gray)
    #plt.show()





