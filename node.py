from collections import Counter
import matplotlib.pyplot as plt
import re
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def default_tokenizer(txt):
    txt = re.sub(r'[^\w]', ' ', txt)
    return txt.lower().split()

def identity_freq(x):
    return x

def log_norm_freq(x, eps=1e-12):
    return math.log(x + eps)

class Node:
    """Encapsulates a counter object to track count, and calculate frequency of words in a set of docs"""
    def __init__(self, documents=[[]], name='anon'):
        self.counter = Counter()
        self.load(documents)
        self.profile = []
        self.name = name
        self.cutoff = 100
        self.depth = 100

    def load(self, documents, uniquify=False):
        assert documents, "missing list of documents, text single doc per line"
        assert isinstance(documents, list), "documents must be list"
        assert isinstance(documents[0], list), "each document is also a list"

        def _get_new_counts(document):
            return Counter(document) if not uniquify else Counter(list(set(document)))

        for idx, document in enumerate(documents):
            new_counter = _get_new_counts(document)
            self.counter.update(new_counter)
            if idx % 1000 == 0:
                print("load: {}\r".format(idx), end='')
        return self

    def load_file(self, fpath, tokenizer=None, uniquify=False):
        """Load a flat file, with one document per line, no header, no csv"""
        assert isinstance(fpath, str), "fpath must be type str not {}".format(type(fpath))
        tokenizer = default_tokenizer if tokenizer is None else tokenizer

        with open(fpath, 'r') as source:
            for idx, line in enumerate(source):
                words = list(set(tokenizer(line))) if uniquify else tokenizer(line)
                self.counter.update(words)
                if idx % 1000 == 0:
                    print(" '{}\r".format(idx), end="")
        return self

    def trim_counter(self, depth):
        """Trim the counter member variable to only the top most common as per depth given"""
        assert isinstance(depth, int), "Depth must be an integer"
        d = {k: v for k, v in self.counter.most_common(depth)}
        self.counter = Counter(d)
        return self

    def merge(self, node, depth=None):
        """Merge the counts in this node with the given node"""
        assert isinstance(node, Node), "Merge node must be type Node"

        self.counter.update(node.counter)
        if depth is not None:
            self.trim_counter(depth)
        return self

    def get_frequencies(self, limit=None):
        """Get normalized frequencies"""
        total = sum(self.counter.values())
        items = self.counter.most_common(limit) if limit is not None else self.counter.most_common()
        return {key: float(val) / total for key, val in items}

    def num_keys(self):
        return len(self.counter.keys())

    def keys_sorted_by_frequency(self, cutoff=100):
        return [key for key, _ in self.counter.most_common()][:cutoff]

    def create_profile(self, node_y, cutoff=100, ratio=0.5):
        _, _, self.profile = self.create_xy_table(node_y, cutoff=cutoff, ratio=ratio)
        return self.profile

    def create_xy_table(self, node2, cutoff=100):
        freq1 = self.get_frequencies(limit=cutoff)
        freq2 = node2.get_frequencies(limit=cutoff)
    
        all_keys = sorted(set(freq1.keys()) | set(freq2.keys()))
    
        x, y, final_keys = [], [], []
        for key in all_keys:
            f1 = freq1.get(key, 0.0)
            f2 = freq2.get(key, 0.0)
            x.append(f1)
            y.append(f2)
            final_keys.append(key)
    
        return x, y, final_keys


    def show_top(self, node2, cutoff=20, ratio=0.5):
        x, y, keys = self.create_xy_table(node2, cutoff=cutoff)
    
        rows = []
        for xi, yi, word in zip(x, y, keys):
            r = math.sqrt(xi**2 + yi**2)
            rows.append((r, xi, yi, word))
    
        rows.sort(reverse=True)
    
        with open(self.name + '.csv', 'w') as target:
            for _, xi, yi, word in rows[:cutoff]:
                target.write("{},{},{}\n".format(str(xi), str(yi), word))
                print("%.4f" % xi, "%.4f" % yi, word)


    def visualize(self, background, num_labeled=10, viz=True, cutoff=100,
                  band_ratio=0.5, black_hole_radius=0.002,
                  ax=None, xmax=None, ymax=None):

        lst_x, lst_y, keys = self.create_xy_table(background, cutoff=cutoff)

        if ax is None:
            fig, ax = plt.subplots()

        max_val = max(lst_x + lst_y) if (lst_x and lst_y) else 1.0
        high = max_val * 1.1
        low = 0.0

        x_high = xmax if xmax is not None else high
        y_high = ymax if ymax is not None else high

        ax.set_xlim(low, x_high)
        ax.set_ylim(low, y_high)
        ax.set_aspect('equal')

        k = 1 + band_ratio
        categories = classify_points(lst_x, lst_y, keys, k=k, black_hole_radius=black_hole_radius)
        colors = [color_map[c] for c in categories]

        ax.scatter(lst_x, lst_y, c=colors)

        # score labels differently by region
        diag_points = []
        x_points = []
        y_points = []

        for xi, yi, w, c in zip(lst_x, lst_y, keys, categories):
            r = math.sqrt(xi**2 + yi**2)

            # never label black-hole words
            if r < black_hole_radius:
                continue

            # if we're in a zoomed panel, only label visible points
            if xi > x_high or yi > y_high:
                continue

            if c == "diagonal":
                # strong + close to diagonal
                score = r - abs(yi - xi)
                diag_points.append((score, xi, yi, w))
            elif c == "x_axis":
                # prefer words strongly biased toward corpus A
                score = xi - yi
                x_points.append((score, xi, yi, w))
            elif c == "y_axis":
                # prefer words strongly biased toward corpus B
                score = yi - xi
                y_points.append((score, xi, yi, w))

        diag_points.sort(reverse=True)
        x_points.sort(reverse=True)
        y_points.sort(reverse=True)

        # separate label budgets per region
        diag_n = num_labeled
        x_n = num_labeled
        y_n = num_labeled

        for _, xi, yi, w in diag_points[:diag_n]:
            ax.annotate(
                w,
                (xi, yi),
                fontsize=9,
                xytext=(3, 3),
                textcoords='offset points'
            )

        for _, xi, yi, w in x_points[:x_n]:
            ax.annotate(
                w,
                (xi, yi),
                fontsize=9,
                xytext=(3, 6),
                textcoords='offset points'
            )

        for _, xi, yi, w in y_points[:y_n]:
            ax.annotate(
                w,
                (xi, yi),
                fontsize=9,
                xytext=(3, 3),
                textcoords='offset points'
            )

        xs = [low, x_high]
        ax.plot(xs, xs, linestyle='--')

        upper = [x * k for x in xs]
        lower = [x / k for x in xs]

        ax.plot(xs, upper, linestyle=':')
        ax.plot(xs, lower, linestyle=':')

        circle = Circle((0, 0), black_hole_radius, fill=False)
        ax.add_patch(circle)

        ax.set_xlabel("Normalized frequency in corpus A")
        ax.set_ylabel("Normalized frequency in corpus B")
        ax.set_title("Word Frequency Geometry")

        if viz and ax is not None and ax.figure:
            plt.show()

    def visualize_dual(self, background, num_labeled=10, cutoff=100,
                       band_ratio=0.5, black_hole_radius=0.002,
                       zoom_max=0.03):

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Full view
        self.visualize(
            background,
            num_labeled=num_labeled,
            viz=False,
            cutoff=cutoff,
            band_ratio=band_ratio,
            black_hole_radius=black_hole_radius,
            ax=axes[0]
        )
        axes[0].set_title("Full View")

        # Zoomed view
        self.visualize(
            background,
            num_labeled=num_labeled,
            viz=False,
            cutoff=cutoff,
            band_ratio=band_ratio,
            black_hole_radius=black_hole_radius,
            ax=axes[1],
            xmax=zoom_max,
            ymax=zoom_max
        )
        axes[1].set_title(f"Zoomed View (0 to {zoom_max})")

        plt.tight_layout()
        plt.show()

def classify_points(x, y, words, k=1.5, black_hole_radius=0.002, eps=1e-12):
    categories = []

    for xi, yi, w in zip(x, y, words):
        r = math.sqrt(xi**2 + yi**2)

        if r < black_hole_radius:
            categories.append("black_hole")
            continue

        ratio = (yi + eps) / (xi + eps)

        if (1/k) <= ratio <= k:
            categories.append("diagonal")
        elif xi > yi:
            categories.append("x_axis")
        else:
            categories.append("y_axis")

    return categories

color_map = {
    "black_hole": "lightgray",
    "diagonal": "blue",
    "x_axis": "red",
    "y_axis": "green"
}


