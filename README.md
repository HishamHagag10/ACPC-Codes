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
# Suffix Array
```CPP
#define vi vector<int>
#define all(x) x.begin(), x.end()
#define sz(x) (int)x.size()
#define rep(i, x, n) for(int i=x;i<n;i++)

struct SuffixArray {
	vi sa, lcp;
	SuffixArray(string s, int lim=256) {
		s.push_back(0); int n = sz(s), k = 0, a, b;
		vi x(all(s)), y(n), ws(max(n, lim));
		sa = lcp = y, iota(all(sa), 0);
		for (int j = 0, p = 0; p < n; j = max(1, j * 2), lim = p) {
			p = j, iota(all(y), n - j);
			rep(i,0,n) if (sa[i] >= j) y[p++] = sa[i] - j;
			fill(all(ws), 0);
			rep(i,0,n) ws[x[i]]++;
			rep(i,1,lim) ws[i] += ws[i - 1];
			for (int i = n; i--;) sa[--ws[x[y[i]]]] = y[i];
			swap(x, y), p = 1, x[sa[0]] = 0;
			rep(i,1,n) a = sa[i - 1], b = sa[i], x[b] =
				(y[a] == y[b] && y[a + j] == y[b + j]) ? p - 1 : p++;
		}
		for (int i = 0, j; i < n - 1; lcp[x[i++]] = k)
			for (k && k--, j = sa[x[i] - 1];
					s[i + k] == s[j + k]; k++);
	}
};
```

# AHO
```cpp
struct AC{
 int N, P;
 const int A = 26;
 vector<int>link, out_link;
 vector<vector<int>>next, out;
 AC(): N(0), P(0) {node();}
 int node(){
     next.emplace_back(A, 0);
     out.emplace_back(0);
     link.emplace_back(0);
     out_link.emplace_back(0);
     return N++;
 }

 inline int get(char c){ return c - 'a'; }

 int add_pattern(const string t){
     int u = 0;
     for(auto &c: t){
         if(!next[u][get(c)])next[u][get(c)] = node();
         u = next[u][get(c)];
     }
     out[u].push_back(P);
     return P++;
 }

 void compute(){
     queue<int>q;
     q.push(0);
     while(!q.empty()){
         int u = q.front();q.pop();
         for(int c = 0; c < A; c++){
             int v = next[u][c];
             if(!v)next[u][c] = next[link[u]][c];
             else {
                 link[v] = u? next[link[u]][c]: 0;
                 out_link[v] = out[link[v]].empty()? out_link[link[v]]: link[v];
                 q.push(v);
             }
         }
     }
 }

 int advance(int u, char c){
     while(u && !next[u][get(c)])u = link[u];
     u = next[u][get(c)];
     return u;
 }
};
```

# Manacher
```CPP
struct Manacher {
  vector<int> p[2];
  // p[1][i] = (max odd length palindrome centered at i) / 2 [floor division]
  // p[0][i] = same for even, it considers the right center
  // e.g. for s = "abbabba", p[1][3] = 3, p[0][2] = 2
  Manacher(string s) {
    int n = s.size();
    p[0].resize(n + 1);
    p[1].resize(n);
    for (int z = 0; z < 2; z++) {
      for (int i = 0, l = 0, r = 0; i < n; i++) {
        int t = r - i + !z;
        if (i < r) p[z][i] = min(t, p[z][l + t]);
        int L = i - p[z][i], R = i + p[z][i] - !z;
        while (L >= 1 && R + 1 < n && s[L - 1] == s[R + 1]) 
          p[z][i]++, L--, R++;
        if (R > r) l = L, r = R;
      }
    }
  }
  bool is_palindrome(int l, int r) {
    int mid = (l + r + 1) / 2, len = r - l + 1;
    return 2 * p[len % 2][mid] + len % 2 >= len;
  }
};
```
# Hashing
```cpp
long long compute_hash(string const& s) {
    const int p = 31;
    const int m = 1e9 + 9;
    long long hash_value = 0;
    long long p_pow = 1;
    for (char c : s) {
        hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
        p_pow = (p_pow * p) % m;
    }
    return hash_value;
}
vector<vector<int>> group_identical_strings(vector<string> const& s) {
    int n = s.size();
    vector<pair<long long, int>> hashes(n);
    for (int i = 0; i < n; i++)
        hashes[i] = {compute_hash(s[i]), i};

    sort(hashes.begin(), hashes.end());

    vector<vector<int>> groups;
    for (int i = 0; i < n; i++) {
        if (i == 0 || hashes[i].first != hashes[i-1].first)
            groups.emplace_back();
        groups.back().push_back(hashes[i].second);
    }
    return groups;
}
int count_unique_substrings(string const& s) {
    int n = s.size();

    const int p = 31;
    const int m = 1e9 + 9;
    vector<long long> p_pow(n);
    p_pow[0] = 1;
    for (int i = 1; i < n; i++)
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(n + 1, 0);
    for (int i = 0; i < n; i++)
        h[i+1] = (h[i] + (s[i] - 'a' + 1) * p_pow[i]) % m;

    int cnt = 0;
    for (int l = 1; l <= n; l++) {
        unordered_set<long long> hs;
        for (int i = 0; i <= n - l; i++) {
            long long cur_h = (h[i + l] + m - h[i]) % m;
            cur_h = (cur_h * p_pow[n-i-1]) % m;
            hs.insert(cur_h);
        }
        cnt += hs.size();
    }
    return cnt;
}

```
# KMP
```cpp
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
	        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi; 
}
```
# KMP Automation
```cpp
vector<int> prefix_function(string s) {
    int n = (int)s.length();
    vector<int> pi(n);
    for (int i = 1; i < n; i++) {
        int j = pi[i-1];
        while (j > 0 && s[i] != s[j])
            j = pi[j-1];
        if (s[i] == s[j])
            j++;
        pi[i] = j;
    }
    return pi; 
}
 
vector<int>pi;
vector<vector<int>>aut, dp;
 
vector<vector<int>>compute_automation(string &s){
    s+="#";
    int n = (int)s.size();
    vector<vector<int>> aut(n, vector<int>(26));
    pi = prefix_function(s);
    aut[0][s[0]-'a'] = 1;
    for(int i = 1; i < n; i++){
        for(int j = 0; j < 26; j++){
            if(s[i] - 'a' == j)
                aut[i][j] = i + 1;
            else
                aut[i][j] = aut[pi[i-1]][j];
        }
    }
    return aut;
}
```
# Z Algorthim
```cpp
vector<int> z_function(string s) {
    int n = s.size();
    vector<int> z(n);
    int l = 0, r = 0;
    for(int i = 1; i < n; i++) {
        if(i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while(i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if(i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }
    return z;
}
```
# Matrix
```cpp
const int mod = 1e9 + 7;

using Row = vector<int>;
using Matrix = vector<Row>;

Matrix mul(Matrix &a, Matrix &b) {
    int n = a.size(), m = a[0].size(), k = b[0].size();
    Matrix res(n, Row(k));
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < k; ++j)
            for(int o = 0; o < m; ++o) {
                res[i][j] += 1ll * a[i][o] * b[o][j] % mod;
                if(res[i][j] >= mod) res[i][j] -= mod;
                if(res[i][j] < 0) res[i][j] += mod;
            }
    return res;
}

Matrix power(Matrix a, int b) {
    int n = a.size();
    Matrix res(n, Row(n));
    for(int i = 0; i < n; ++i) res[i][i] = 1;

    while(b) {
        if(b&1) res = mul(res, a);
        a = mul(a, a), b >>= 1;
    }

    return res;
}
```
# MO
```CPP
struct Query {
    int l, r, ind;
    pair<int, int> toPair() const {
        return make_pair(l / BLOCK_SIZE, ((l / BLOCK_SIZE) & 1) ? -r : +r);
    }
};
bool operator<(const Query &a, const Query &b) {
    return a.toPair() < b.toPair();
}

ll n, Q, a[N], ans[N], freq[N], cur = 0, BLOCK_SIZE = sqrt(N) + 1;
Query q[N];

void add(int ind) {}
void remove(int ind) {}

void MO_PROCESS(){
    int l = 1, r = 0;
    sort(q, q + Q);
    for (int i = 0; i < Q; i++){
        while (q[i].l < l) add(--l);
        while (q[i].l > l) remove(l++);
        while (q[i].r < r) remove(r--);
        while (q[i].r > r) add(++r);
        ans[q[i].ind] = cur;
    }
}

void solve() {
    cin >> n >> Q;
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < Q; i++){
        int l, r; cin >> l >> r; l--; r--;
        q[i] = {l, r, i};
    }
    MO_PROCESS();
    for (int i = 0; i < Q; i++) cout << ans[i] << "\n";
}
```
# MO (helbirt)
```cpp
const int N = 1e6 + 10;

int n, Q;
int a[N], res[N], freq[N], cur = 0;

inline int64_t hilbertOrder(int x, int y, int pow, int rotate) {
    if (pow == 0) return 0;
    int hpow = 1 << (pow - 1);
    int seg = (x < hpow) ? ((y < hpow) ? 0 : 3) : ((y < hpow) ? 1 : 2);
    seg = (seg + rotate) & 3;
    const int rotateDelta[4] = {3, 0, 0, 1};
    int nx = x & (x ^ hpow), ny = y & (y ^ hpow);
    int nrot = (rotate + rotateDelta[seg]) & 3;
    int64_t subSquareSize = 1LL << (2 * pow - 2);
    int64_t ans = seg * subSquareSize;
    int64_t add = hilbertOrder(nx, ny, pow - 1, nrot);
    ans += (seg == 1 || seg == 2) ? add : (subSquareSize - add - 1);
    return ans;
}

struct Query {
    int l, r, idx;
    int64_t ord;
    void calcOrder() {
        ord = hilbertOrder(l, r, 21, 0);
    }
};

bool operator<(const Query &x, const Query &y) {
    return x.ord < y.ord;
}

Query q[N];

void add(int ind) {
    
}

void remove(int ind) {
    
}

void MO_PROCESS() {
    for (int i = 0; i < Q; ++i)
        q[i].calcOrder();
    sort(q, q + Q);
    int l = 0, r = -1;
    for (int i = 0; i < Q; i++) {
        while (l > q[i].l) add(--l);
        while (r < q[i].r) add(++r);
        while (l < q[i].l) remove(l++);
        while (r > q[i].r) remove(r--);
        res[q[i].idx] = cur;
    }
}

void solve() {
    cin >> n >> Q;
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < Q; i++) {
        int l, r;
        cin >> l >> r;
        l--, r--;
        q[i] = {l, r, i};
    }

    MO_PROCESS();

    for (int i = 0; i < Q; i++)
        cout << res[i] << "\n";
}
```
# DSU 
```cpp
struct DSU
{
    vector<int> p, rnk;
    DSU(int n) { p = vector<int>(n + 1, -1); rnk = vector<int>(n + 1, 0); }
    int size(int x) { return -p[find_set(x)]; }
    bool sameSet(int a, int b) { return find_set(a) == find_set(b); }
    int find_set(int x) { return (p[x] < 0) ? x : p[x] = find_set(p[x]); }
    bool union_set(int a, int b)
    {
        a = find_set(a);
        b = find_set(b);
        if (a == b) return false;
        if (rnk[b] > rnk[a]) swap(a, b);
        rnk[a] += (rnk[a] == rnk[b]);
        p[a] += p[b];
        p[b] = a;
        return true;
    }
};
```
# DSU bipartite
```cpp
struct DSU {
    vector<int> rnk;
    vector<pii> p;
    vector<bool> bipartite;
    DSU(int n) {
        p = vector<pii>(n + 1, {-1, 0});
        rnk = vector<int>(n + 1, 0);
        bipartite = vector<bool>(n + 1, true);
    }
    int size(int x) { return -p[find_set(x).first].first; }
    bool sameSet(int a, int b) { return find_set(a) == find_set(b); }
    pii find_set(int x) {
        if (p[x].first < 0) return {x, p[x].second};
        int parity = p[x].second;
        p[x] = find_set(p[x].first);
        p[x].second ^= parity;
        return p[x];
    }
    bool union_set(int a, int b) {
        pii x = find_set(a), y = find_set(b);
        if (x.first == y.first) {
            if (x.second == y.second) bipartite[x.first] = false;
            return false;
        }
        if (rnk[y.first] > rnk[x.first]) swap(x, y);
        rnk[x.first] += (rnk[x.first] == rnk[y.first]);
        p[x.first].first += p[y.first].first;
        p[y.first] = {x.first, x.second ^ y.second ^ 1};
        bipartite[x.first] = bipartite[x.first] && bipartite[y.first];
        return true;
    }
    bool is_bipartite(int x) {
        return bipartite[find_set(x).first];
    }
};
```
# Floyed
```CPP
#include <iostream>
#include <vector>
#define ll long long
constexpr ll OO = 1e18;
using namespace std;
int main() {
    ll n, m, q, u, v, d;
    std::cin >> n >> m >> q;
    vector<vector<ll>> min_dis(n, vector<ll>(n, OO));
    for (int i = 0; i < m; i++) {
        std::cin >> u >> v >> d;
        u--; v--;
        min_dis[u][v] = min(min_dis[u][v], d);
        min_dis[v][u] = min(min_dis[v][u], d);
    }
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (min_dis[i][k] < OO && min_dis[k][j] < OO)
                    min_dis[i][j] = min(min_dis[j][i], min_dis[i][k] + min_dis[k][j]);
    while (q--) {
        std::cin >> u >> v;
        u--; v--;
        if (u == v) std::cout << "0\n";
        else std::cout << ((min_dis[u][v] == OO) ? -1 : min_dis[u][v]) << "\n";
    }
}
```
# Targian (SCC)
```CPP
#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define MOD 1000000007
const int N = 1e6 + 10;
using namespace std;
vector<vector<int>> adj, adj_rev, scc;
vector<int> order, component;
bitset<N> vis;
void dfs1(int cur) {
    vis[cur] = 1;
    for (auto i : adj[cur])
        if (!vis[i]) dfs1(i);
    order.push_back(cur);
}
void dfs2(int cur) {
    vis[cur] = 1;
    component.push_back(cur);
    for (auto i : adj_rev[cur])
        if (!vis[i]) dfs2(i);
}
int main() {
    int n, m;
    cin >> n >> m;
    adj.assign(n + 1, vector<int>());
    adj_rev.assign(n + 1, vector<int>());
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj_rev[v].push_back(u);
    }
    for (int i = 1; i <= n; i++) {
        if (!vis[i]) dfs1(i);
    }
    vis.reset();
    reverse(order.begin(), order.end());
    for (auto i : order) {
        if (vis[i]) continue;
        dfs2(i);
        scc.push_back(component);
        component.clear();
    }
    cout << scc.size() << "\n";
    for (auto com : scc) {
        cout << "Component: ";
        for (auto i : com) cout << i << " ";
        cout << "\n";
    }
}
```
# Bridge
```cpp
vector<vector<int>>adj;
vector<pii>bridges;
bitset<N>vis;
vector<int>disc,low;
int cur_time=0;
void dfs(int cur,int par){
   vis[cur]=1;
   disc[cur]=low[cur]=cur_time++;
   int cnt=0;
   for(auto i:adj[cur]){
       if(i==par)continue;
       if(!vis[i]){
           dfs(i,cur);
           low[cur]=min(low[cur],disc[i]);
           if(low[i]>disc[cur])bridges.push_back({cur,i});
       }else low[cur]=min(low[cur],disc[i]);
   }
}
int main() {
   int n,m;cin>>n>>m;
   adj.assign(n+1,vector<int>());
   disc.assign(n+1,-1);
   low.assign(n+1,-1);
   for(int i=0;i<m;i++){
       int u,v;cin>>u>>v;
       adj[u].push_back(v);
       adj[v].push_back(u);
   }
   dfs(1,0);
   cout<<bridges.size()<<"\n";
   for(auto [x,y]:bridges)cout<<x<<" "<<y<<"\n";
}
```

# DP patterns
## LIS
```CPP
int LIS(vector<int>&arr)
{
    int n=arr.size(),ans=0;
    vector<int>dp;
    dp.push_back(arr[0]);
    for(int i=1;i<n;i++){
        if(arr[i]>dp.back())
            dp.push_back(arr[i]);
        else{
            int l=(lower_bound(dp.begin(),dp.end(),arr[i])-dp.begin());
            if(dp[l-1]<arr[i] && arr[i]<dp[l])
                dp[l]=arr[i];
        }
    }
    return dp.size();
}
```
## LIS and retrieve sequence 
```cpp
vector<int>LIS(vector<int>arr,int n){
    vector<int>seq,dp;
    map<int,int>p;
    for(int i=0;i<n;i++){
        int l=lower_bound(dp.begin(),dp.end(),arr[i])-dp.begin();
        if(l==dp.size())dp.push_back(arr[i]);
        else dp[l]=arr[i];
        if(l)p[arr[i]]=dp[l-1];
    }
    for(int cur=(int)dp.back();1;cur=p[cur]){
        seq.push_back(cur);
        if(!p.count(cur))break;
    }
    reverse(seq.begin(),seq.end());
    return seq;
}
```
## DP Digit
- Statement: How many numbers x are there in the range a to b, where the digit d occurs exactly k times in x?
```cpp
int d,k,dp[20][20][2];
int calc(string num,int pos,int cnt,bool yes){
    if(cnt>k)return 0;
    if(pos==num.size())return cnt==k;
    int &ret=dp[pos][cnt][yes];
    if(~ret)return ret;
    int lmt=(yes)?9:(num[pos]-'0');
    ret=0;
    for(int i=0;i<=lmt;i++){
        ret+=calc(num,pos+1,cnt+(d==i),yes || i<lmt);
    }
    return ret;
}
void solve(){
    int a,b;cin>>a>>b>>k>>d;
    memset(dp,-1,sizeof(dp));
    int x=calc(to_string(b),0,0,0);
    memset(dp,-1,sizeof(dp));
    int y=calc(to_string(a-1),0,0,0);
    cout<<x-y<<"\n";
}
```
## 
- - Problem: [G. Maximum Product]([https://codeforces.com/problemset/gymProblem/100886/G](https://codeforces.com/problemset/gymProblem/100886/G "https://codeforces.com/problemset/gymProblem/100886/G (Ctrl or Cmd-click to open)"))
- Statement : Find the number from the range [a, b] which has the maximum product of the digits.
```cpp
string a,b;
ll dp[20][2][2][2];
ll calc(ll pos,bool above,bool under,bool lz){
    if(pos==a.size())return 1;
    ll &ret=dp[pos][above][under][lz];
    if(~ret)return ret;
    ll start=(above)?0:a[pos]-'0',end=(under)?9:b[pos]-'0';
    for(ll i=start;i<=end;i++){
        ll cur=calc(pos+1,above || i>start,under || i<end,lz || i);
        if(i || lz)cur*=i;
        if(ret<cur)ret=cur;
    }
    return ret;
}
string ans="";
void build(ll pos,bool above,bool under,bool lz){
    if(pos==a.size())return;
    ll start=(above)?0:a[pos]-'0',end=(under)?9:b[pos]-'0';
    ll mx=-1,ind=-1;
    for(ll i=start;i<=end;i++){
        ll cur=calc(pos+1,above || i>start,under || i<end,lz || i);
        if(i || lz)cur*=i;
        if(mx<cur){
            mx=cur;
            ind=i;
        }
    }
    if(ind || lz)
        ans+=to_string(ind);
    build(pos+1,above || ind>start,under || ind<end,lz || ind);
}
int main(){
    cin>>a>>b;
    while(a.size()!=b.size())a='0'+a;
    memset(dp,-1,sizeof dp);
    calc(0,0,0,0);
    build(0,0,0,0);
    cout<<ans<<"\n";
}
```
## 
- Statement : Your task is to count the number of integers between a and b where no two adjacent digits are the same.
```CPP
string a,b;
ll dp[20][11][2][2][2];
ll calc(ll pos,ll prev,ll above,ll under,ll lz){
    if(pos==b.size())return 1;
    ll &ret=dp[pos][prev][above][under][lz];
    if(~ret)return ret;
    ll start=(above)?0:a[pos]-'0',end=(under)?9:b[pos]-'0';
    ret=0;
    for(ll i=start;i<=end;i++){
        if(i==prev)continue;
        if(lz || i)ret+=calc(pos+1,i,above || i>start,under || i<end,lz || i);
        else ret+=calc(pos+1,10,above || i>start,under || i<end,lz || i);
    }
    return ret;
}
int main(){
    cin>>a>>b;
    memset(dp,-1,sizeof(dp));
    while(a.size()!=b.size())a='0'+a;
    cout<<calc(0,10,0,0,0)<<"\n";
}
```

## Kadane's Algorithm
- Statement: Given an array of integers, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
```cpp
ll Kadane_Sum(vector<ll>& nums) {
    ll local_max=-INF,global_max=-INF;
    for(int i=0;i<nums.size();i++)
    {
        local_max=max(nums[i],local_max+nums[i]);
        global_max=max(global_max,local_max);
    }
    return global_max;
}
```
## DP Masking
- Problem: [A. Gym Plates](https://codeforces.com/gym/104493/problem/A)
- Statement: used when you want to know if each element appear at most 1 --> N=20 (n is very small) what if you want to know if element appear at most 2 , you can use base 2 instead of base 2

1. make a vector represent representation of each number(in my range) in 3 base representation
```cpp
void pre(){    
    mp.assign(M,vector<int>());
    for(int i=0;i<N;i++){
        ll x=i;
        vector<int>bit(11);
        int ind=0;
        while(x){
            bit[ind++]=x%3;
            x/=3;
        }
        mp[i]=bit;   // expample : mp[3] = 1 0 0...
    }
} 
```
2. to get the new mask after adding new element
```cpp
int power=1,new_mask=0;
for(int i=0;i<10;i++){
    new_mask+=power*bit[i];
    power*=3;
}
```

## NCR
```cpp
// A Lucas Theorem based solution to compute nCr % p
#include<bits/stdc++.h>
using namespace std;

// Returns nCr % p.  In this Lucas Theorem based program,
// this function is only called for n < p and r < p.
int nCrModpDP(int n, int r, int p)
{
    // The array C is going to store last row of
    // pascal triangle at the end. And last entry
    // of last row is nCr
    int C[r+1];
    memset(C, 0, sizeof(C));

    C[0] = 1; // Top row of Pascal Triangle

    // One by constructs remaining rows of Pascal
    // Triangle from top to bottom
    for (int i = 1; i <= n; i++)
    {
        // Fill entries of current row using previous
        // row values
        for (int j = min(i, r); j > 0; j--)

            // nCj = (n-1)Cj + (n-1)C(j-1);
            C[j] = (C[j] + C[j-1])%p;
    }
    return C[r];
}

// Lucas Theorem based function that returns nCr % p
// This function works like decimal to binary conversion
// recursive function.  First we compute last digits of
// n and r in base p, then recur for remaining digits
int nCrModpLucas(int n, int r, int p)
{
   // Base case
   if (r==0)
      return 1;

   // Compute last digits of n and r in base p
   int ni = n%p, ri = r%p;

   // Compute result for last digits computed above, and
   // for remaining digits.  Multiply the two results and
   // compute the result of multiplication in modulo p.
   return (nCrModpLucas(n/p, r/p, p) * // Last digits of n and r
           nCrModpDP(ni, ri, p)) % p;  // Remaining digits
}

// Driver program
int main()
{
    int n = 1000, r = 900, p = 13;
    cout << "Value of nCr % p is " << nCrModpLucas(n, r, p);
    return 0;
}
```
