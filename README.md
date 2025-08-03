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


# HLD (Nodes start from 0)
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
# Kth Ancestor (from lca)
```C++
const int MAX_N = 100005;
const int LOG = 20;
vector<int> children[MAX_N];
int up[MAX_N][LOG]; // up[v][j] is 2^j-th ancestor of v
int depth[MAX_N], sons[MAX_N];
int dfs(int a) {
    sons[a] = 1; // count itself
	for(int b : children[a]) {
	    if (up[a][0]==b)continue;
		depth[b] = depth[a] + 1;
		up[b][0] = a; // a is parent of b
		for(int j = 1; j < LOG; j++) {
			up[b][j] = up[up[b][j-1]][j-1];
		}
		sons[a] += dfs(b);
	}
	return sons[a];
}
 
int get_lca(int a, int b) { // O(log(N))
	if(depth[a] < depth[b]) {
		swap(a, b);
	}
	// 1) Get same depth.
	int k = depth[a] - depth[b];
    int ret = k;
    // cout<<k<<'\n';
	for(int j = LOG - 1; j >= 0; j--) {
		if(k & (1 << j)) {
			a = up[a][j]; // parent of a
		}
	}
    // cout<<a<<' '<<b<<'\n';
	// 2) if b was ancestor of a then now a==b
	if(a == b) {
		return ret;
	}
	// 3) move both a and b with powers of two
	for(int j = LOG - 1; j >= 0; j--) {
		if(up[a][j] != up[b][j]) {
			a = up[a][j];
			b = up[b][j];
		    // cout<<a<<' '<<j<<'\n';
            ret += (1 << j) * 2; // count the steps taken
		}
	}
    // cout<<up[a][0]<<'\n';
    return ret + 2; // return the distance
	return up[a][0];
}
int getKthAncestor(int node, int k) {
    for (int i = 0; i < 20; i++) {
        if ((k >> i) & 1) {
            node = up[node][i];
            if (node == -1) return -1;
        }
    }
    return node;
}
```
# Modulo of Fraction 
```C++
long long power(long long base, long long exp, long long mod) {
    if (exp == 0) return 1;
    if (exp % 2 == 0) {
        long long half_pow = power(base, exp / 2, mod);
        return (half_pow * half_pow) % mod;
    } else {
        return (base * power(base, exp - 1, mod)) % mod;
    }
}

// Function to compute the modular inverse of 'a' modulo 'm' using Fermat's Little Theorem
long long inverse_modulo(long long a, long long m) {
    return power(a, m - 2, m);
}

// Function to compute the value of (A/B) % m, where A, B, and m are given integers
long long fraction_to_natural_modulo(long long A, long long B, long long m) {
    // Calculate the modular inverse of B
    long long inverse_B = inverse_modulo(B, m);
    // Compute (A * inverse_B) % m
    long long result = (A * inverse_B) % m;
    return result;
}
```
# SPF (prime factroization in log(n))
```C++
#define MAXN 100001
vector<int> spf(MAXN + 1, 1);

// Calculating SPF (Smallest Prime Factor) for every
// number till MAXN.
// Time Complexity : O(nloglogn)
void sieve()
{
    // stores smallest prime factor for every number

    spf[0] = 0;
    for (int i = 2; i <= MAXN; i++) {
        if (spf[i] == 1) { // if the number is prime ,mark
                           // all its multiples who havent
                           // gotten their spf yet
            for (int j = i; j <= MAXN; j += i) {
                if (spf[j]== 1) // if its smallest prime factor is
                          // 1 means its spf hasnt been
                          // found yet so change it to i
                    spf[j] = i;
            }
        }
    }
}

// A O(log n) function returning primefactorization
// by dividing by smallest prime factor at every step
vector<int> getFactorization(int x)
{
    vector<int> ret;
    while (x != 1) {
        ret.push_back(spf[x]);
        x = x / spf[x];
    }
    return ret;
}
```
# Suffix
# AHO
# Macher
# Hashing
# KMP
# KMP Automation
# Z Algorthim
# Matrix
# MO (helbirt)
# DSU biprtite
# Floyed
# Targian (SCC)
# Bridge
# DP patterns
