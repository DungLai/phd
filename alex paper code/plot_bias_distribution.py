# bias_distribution = [1000,800,300,400,200,1000,100,200,50,700]
# bias_distribution = [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
bias_distribution = [200,300,600,900,550,150,800,150,600,40]

# Draw the bias distribution
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
y_pos = np.arange(len(objects))

plt.bar(y_pos, bias_distribution, align='center', color="green")
plt.xticks(y_pos, objects)
plt.ylabel('Number of datapoints')
plt.xlabel('Categories')

plt.title('Custom bias distribution of test dataset')

plt.show()

bias_distribution1 = [1000,800,300,400,200,1000,100,200,50,700]
bias_distribution2 = [200,300,600,900,550,150,800,150,600,40]
labels = [0,1,2,3,4,5,6,7,8,9]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, bias_distribution1, width, label='Bias dataset 1', color = "red")
rects2 = ax.bar(x + width/2, bias_distribution2, width, label='Bias dataset 2', color = "green")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Datapoints count')
ax.set_xlabel('Category')
ax.set_title('Compare 2 bias distributions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()