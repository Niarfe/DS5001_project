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
        _, _, self.profile = self.create_xy_table(node_y, cutoff=cutoff)
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


    def visualize(self, background, num_labeled=10, viz=True,
                  cutoff=100, band_ratio=1.0, axis_ratio=4.0,
                  black_hole_radius=0.002, ax=None, xmax=None,
                  ymax=None):
    
        lst_x, lst_y, keys = self.create_xy_table(background, cutoff=cutoff)

        #% TEMP FIX
        import math
        
        global_rank = []
        for xi, yi, w in zip(lst_x, lst_y, keys):
            r = math.sqrt(xi**2 + yi**2)
            global_rank.append((r, xi, yi, w))
        
        global_rank.sort(reverse=True)
        top_global = [w for _, _, _, w in global_rank[:5]]
        #% END TEMP FIX
    
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
    
        k_diag = 1 + band_ratio
        k_axis = axis_ratio
    
        categories = classify_points(
            lst_x, lst_y, keys,
            k_diag=k_diag,
            k_axis=k_axis,
            black_hole_radius=black_hole_radius
        )
        colors = [color_map[c] for c in categories]
    
        ax.scatter(lst_x, lst_y, c=colors)
    
        # score labels differently by region
        diag_points = []
        x_axis_points = []
        y_axis_points = []
        x_wedge_points = []
        y_wedge_points = []
        black_hole_points = []
    
        for xi, yi, w, c in zip(lst_x, lst_y, keys, categories):
            r = math.sqrt(xi**2 + yi**2)

            if xi > x_high or yi > y_high:
                continue
    
            if c == "black_hole":
                score = r
                black_hole_points.append((score, xi, yi, w))
                continue
        
            ratio = (yi + 1e-12) / (xi + 1e-12)
    
            if c == "diagonal":
                score = (3.0 * r) - abs(math.log(ratio))
                diag_points.append((score, xi, yi, w))
            elif c == "x_axis":
                score = xi - yi
                x_axis_points.append((score, xi, yi, w))
            elif c == "y_axis":
                score = yi - xi
                y_axis_points.append((score, xi, yi, w))
            elif c == "x_wedge":
                score = (xi - yi) * r
                x_wedge_points.append((score, xi, yi, w))
            elif c == "y_wedge":
                score = (yi - xi) * r
                y_wedge_points.append((score, xi, yi, w))

    
        diag_points.sort(reverse=True)
        x_axis_points.sort(reverse=True)
        y_axis_points.sort(reverse=True)
        x_wedge_points.sort(reverse=True)
        y_wedge_points.sort(reverse=True)
        black_hole_points.sort()
    
        diag_n = num_labeled
        x_axis_n = num_labeled
        y_axis_n = num_labeled
        x_wedge_n = num_labeled
        y_wedge_n = num_labeled
        black_hole_n = 1
    
        #%for _, xi, yi, w in diag_points[:diag_n]:
        #%    ax.annotate(w, (xi, yi), fontsize=9, xytext=(3, 3), textcoords='offset points')
        #% TEMP FIX 2
        labeled_words = set()
        
        for _, xi, yi, w in black_hole_points[:black_hole_n]:
            ax.annotate(
                w,
                (xi, yi),
                fontsize=8,
                xytext=(3, -8),
                textcoords='offset points'
            )

        # First: force top global words (like "the")
        for w in top_global:
            for xi, yi, word, c in zip(lst_x, lst_y, keys, categories):
                if word == w:
                    ax.annotate(word, (xi, yi), fontsize=10,
                                xytext=(3, 3), textcoords='offset points')
                    labeled_words.add(word)
                    break
        
        # Then: normal diagonal labeling (skip duplicates)
        for _, xi, yi, w in diag_points:
            if w in labeled_words:
                continue
            ax.annotate(w, (xi, yi), fontsize=9,
                        xytext=(3, 3), textcoords='offset points')
            labeled_words.add(w)
            if len(labeled_words) >= diag_n:
                break
        #% END TEMP FIX 2
    
        for _, xi, yi, w in x_axis_points[:x_axis_n]:
            ax.annotate(w, (xi, yi), fontsize=9, xytext=(3, 6), textcoords='offset points')
    
        for _, xi, yi, w in y_axis_points[:y_axis_n]:
            ax.annotate(w, (xi, yi), fontsize=9, xytext=(3, 3), textcoords='offset points')
    
        for _, xi, yi, w in x_wedge_points[:x_wedge_n]:
            ax.annotate(w, (xi, yi), fontsize=9, xytext=(3, 6), textcoords='offset points')
    
        for _, xi, yi, w in y_wedge_points[:y_wedge_n]:
            ax.annotate(w, (xi, yi), fontsize=9, xytext=(3, 3), textcoords='offset points')
    
        xs = [low, x_high]
    
        # main diagonal
        ax.plot(xs, xs, linestyle='--')
    
        # stop-word band
        upper_diag = [x * k_diag for x in xs]
        lower_diag = [x / k_diag for x in xs]
        ax.plot(xs, upper_diag, linestyle=':')
        ax.plot(xs, lower_diag, linestyle=':')
    
        # wedge/axis boundaries
        upper_axis = [x * k_axis for x in xs]
        lower_axis = [x / k_axis for x in xs]
        ax.plot(xs, upper_axis, linestyle='-.')
        ax.plot(xs, lower_axis, linestyle='-.')
    
        circle = Circle((0, 0), black_hole_radius, fill=False)
        ax.add_patch(circle)
    
        ax.set_xlabel("Normalized frequency in corpus A")
        ax.set_ylabel("Normalized frequency in corpus B")
        ax.set_title("Word Frequency Geometry")
    
        if viz and ax is not None and ax.figure:
            plt.show()


    def visualize_dual(self, background, num_labeled=10, cutoff=100,
                       band_ratio=0.5, axis_ratio=4.0,
                       black_hole_radius=0.002, zoom_max=0.03):
    
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
        self.visualize(
            background,
            num_labeled=num_labeled,
            viz=False,
            cutoff=cutoff,
            band_ratio=band_ratio,
            axis_ratio=axis_ratio,
            black_hole_radius=black_hole_radius,
            ax=axes[0]
        )
        axes[0].set_title("Full View")
    
        self.visualize(
            background,
            num_labeled=num_labeled,
            viz=False,
            cutoff=cutoff,
            band_ratio=band_ratio,
            axis_ratio=axis_ratio,
            black_hole_radius=black_hole_radius,
            ax=axes[1],
            xmax=zoom_max,
            ymax=zoom_max
        )
        axes[1].set_title(f"Zoomed View (0 to {zoom_max})")
    
        plt.tight_layout()
        plt.show()

def classify_points(x, y, words,
                    k_diag=1.5,
                    k_axis=4.0,
                    black_hole_radius=0.002,
                    eps=1e-12):
    categories = []

    for xi, yi, _ in zip(x, y, words):
        r = math.sqrt(xi**2 + yi**2)

        if r < black_hole_radius:
            categories.append("black_hole")
            continue

        ratio = (yi + eps) / (xi + eps)

        # stop-word band
        if (1 / k_diag) <= ratio <= k_diag:
            categories.append("diagonal")

        # strong corpus A keywords
        elif ratio < (1 / k_axis):
            categories.append("x_axis")

        # strong corpus B keywords
        elif ratio > k_axis:
            categories.append("y_axis")

        # style/fingerprint wedges
        elif ratio < (1 / k_diag):
            categories.append("x_wedge")
        else:
            categories.append("y_wedge")

    return categories

color_map = {
    "black_hole": "black",
    "diagonal": "blue",
    "x_axis": "red",
    "y_axis": "green",
    "x_wedge": "orange",
    "y_wedge": "purple",
}

def top_words_by_region(x, y, words, categories,
                        black_hole_radius=0.002,
                        n=10,
                        eps=1e-12):
    buckets = {
        "diagonal": [],
        "x_axis": [],
        "y_axis": [],
        "x_wedge": [],
        "y_wedge": [],
        "black_hole": [],
    }

    for xi, yi, w, c in zip(x, y, words, categories):
        r = math.sqrt(xi**2 + yi**2)
        ratio = (yi + eps) / (xi + eps)

        if c == "black_hole":
            score = -r
        elif c == "diagonal":
            score = (5.0 * r) - abs(math.log(ratio))
        elif c == "x_axis":
            score = xi - yi
        elif c == "y_axis":
            score = yi - xi
        elif c == "x_wedge":
            score = (xi - yi) * r
        elif c == "y_wedge":
            score = (yi - xi) * r
        else:
            score = 0.0

        buckets[c].append((score, xi, yi, w))

    for region in buckets:
        buckets[region].sort(reverse=True)

    return {
        region: [(xi, yi, w) for _, xi, yi, w in rows[:n]]
        for region, rows in buckets.items()
    }
