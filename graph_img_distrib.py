import os
import matplotlib.pyplot as plt

DATA_DIR = 'data/wikiart'

def get_counts(data_path=DATA_DIR):
    counts = {}
    for style in os.listdir(DATA_DIR):
        if os.path.isdir(os.path.join(DATA_DIR, style)):
            counts[style] = len(os.listdir(os.path.join(DATA_DIR, style)))
    return counts

def draw():
    counts = get_counts()
    styles = []
    style_counts = []
    for style, cnt in counts.items():
        styles.append(style)
        style_counts.append(cnt)

    # plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    y_pos = range(len(styles))

    ax.barh(y_pos, style_counts, label='wikiart', align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(styles)
    # fig.legend(loc='upper right')
    ax.invert_yaxis()  # labels read top-to-bottom
    # label bars with counts
    for i, v in enumerate(style_counts):
        ax.text(v + 3, i + .25, str(v), color='black')
    ax.set_xlabel('Number of images')
    ax.set_title('Distribution of images to art styles')
    plt.xlim((0,15000))
    plt.tight_layout()
    # plt.show()
    fig.savefig('assets/img_distrib.png', dpi = 300)   

if __name__ == "__main__":
    draw()
