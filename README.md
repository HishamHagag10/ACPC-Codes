# ACPC-Codes

# Segment Tree Code with lazy to find max sum subarray 
```C++
struct Node {
    long long max_sum;    // Maximum subsegment sum
    long long prefix_sum; // Maximum prefix sum
    long long suffix_sum; // Maximum suffix sum
    long long total_sum;  // Total sum of the segment

    Node(long long ms = 0, long long ps = 0, long long ss = 0, long long ts = 0)
        : max_sum(ms), prefix_sum(ps), suffix_sum(ss), total_sum(ts) {}
};

class SegmentTree {
private:
    int n;
    vector<Node> tree;
    vector<long long> lazy;
    vector<bool> lazy_exists; // Tracks if a lazy update is pending



    void build(const vector<long long>& arr, int node, int start, int end) {
        if (start == end) {
            long long val = arr[start];
            tree[node] = Node(val, val, val, val);
            return;
        }
        int mid = (start + end) / 2;
        build(arr, 2 * node + 1, start, mid);
        build(arr, 2 * node + 2, mid + 1, end);
        tree[node] = merge(tree[2 * node + 1], tree[2 * node + 2]);
    }

    void propagate(int node, int start, int end) {
        if (lazy_exists[node]) {
            long long val = lazy[node];
            // Set the node's values as if all elements in the range are 'val'
            tree[node].max_sum = max(0ll,(end-start+1)*val);
            tree[node].prefix_sum = (end-start+1)*val;
            tree[node].suffix_sum = (end-start+1)*val;
            tree[node].total_sum = val * (end - start + 1);
            if (start != end) {
                // Propagate the set operation to children
                lazy[2 * node + 1] = val;
                lazy[2 * node + 2] = val;
                lazy_exists[2 * node + 1] = true;
                lazy_exists[2 * node + 2] = true;
            }
            lazy_exists[node] = false;
        }
    }

public:
    Node merge(const Node& left, const Node& right) {
        Node result;
        result.total_sum = left.total_sum + right.total_sum;
        result.prefix_sum = max(left.prefix_sum, left.total_sum + right.prefix_sum);
        result.suffix_sum = max(right.suffix_sum, right.total_sum + left.suffix_sum);
        result.max_sum = max({left.max_sum, right.max_sum, left.suffix_sum + right.prefix_sum});
        return result;
    }
    SegmentTree(const vector<long long>& arr) {
        n = arr.size();
        tree.resize(8 * n);
        lazy.resize(8 * n, 0);
        lazy_exists.resize(8 * n, false);
        build(arr, 0, 0, n - 1);
    }

    void update_range(int left, int right, long long new_value) {
        update_range(0, 0, n - 1, left, right, new_value);
    }

private:
    void update_range(int node, int start, int end, int left, int right, long long new_value) {
        propagate(node, start, end);
        if (start > right || end < left) return;
        if (left <= start && end <= right) {
            lazy[node] = new_value;
            lazy_exists[node] = true;
            propagate(node, start, end);
            return;
        }
        int mid = (start + end) / 2;
        update_range(2 * node + 1, start, mid, left, right, new_value);
        update_range(2 * node + 2, mid + 1, end, left, right, new_value);
        tree[node] = merge(tree[2 * node + 1], tree[2 * node + 2]);
    }

public:
    Node query(int left, int right) {
        return query(0, 0, n - 1, left, right);
    }

private:
    Node query(int node, int start, int end, int left, int right) {
        propagate(node, start, end);
        if (start > right || end < left) return Node();
        if (left <= start && end <= right) return tree[node];
        int mid = (start + end) / 2;
        Node left_node = query(2 * node + 1, start, mid, left, right);
        Node right_node = query(2 * node + 2, mid + 1, end, left, right);
        return merge(left_node, right_node);
    }
};
```
