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


# HLD 
``` C++
vector<int> parent, depth, heavy, head, pos;
int cur_pos;

int dfs(int v, vector<vector<int>> const& adj) {
    int size = 1;
    int max_c_size = 0;
    for (int c : adj[v]) {
        if (c != parent[v]) {
            parent[c] = v, depth[c] = depth[v] + 1;
            int c_size = dfs(c, adj);
            size += c_size;
            if (c_size > max_c_size)
                max_c_size = c_size, heavy[v] = c;
        }
    }
    return size;
}

void decompose(int v, int h, vector<vector<int>> const& adj) {
    head[v] = h, pos[v] = cur_pos++;
    if (heavy[v] != -1)
        decompose(heavy[v], h, adj);
    for (int c : adj[v]) {
        if (c != parent[v] && c != heavy[v])
            decompose(c, c, adj);
    }
}

void init(vector<vector<int>> const& adj) {
    int n = adj.size();
    parent = vector<int>(n);
    depth = vector<int>(n);
    heavy = vector<int>(n, -1);
    head = vector<int>(n);
    pos = vector<int>(n);
    cur_pos = 0;

    dfs(0, adj);
    decompose(0, 0, adj);
}
int query(int a, int b) {
    int res = 0;
    for (; head[a] != head[b]; b = parent[head[b]]) {
        if (depth[head[a]] > depth[head[b]])
            swap(a, b);
        int cur_heavy_path_max = segment_tree_query(pos[head[b]], pos[b]);
        res = max(res, cur_heavy_path_max);
    }
    if (depth[a] > depth[b])
        swap(a, b);
    int last_heavy_path_max = segment_tree_query(pos[a], pos[b]);
    res = max(res, last_heavy_path_max);
    return res;
}
```


# Fast Segment tree (Iterative)
```C++
int n, seg[6*N]={};
int val[N];

void update(int k, int x) {
    k += N; seg[k] = x; k >>= 1;
    while (k > 0) {
        seg[k] = max(seg[2*k], seg[2*k+1]);
        k >>= 1;
    }
}

int query(int a, int b) {
    a += N, b += N;
    int s = 0;
    while (a <= b) {
        if (a & 1) {
            s = max(s, seg[a]);
            a++;
        }
        if (~b & 1) {
            s = max(s, seg[b]);
            b--;
        }
        a >>= 1, b >>= 1;
    }
    return s;
}
```
