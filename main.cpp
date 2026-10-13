// Dijkstra vs A* on BFS subgraphs of the DIMACS BAY road network.
//
// Build:  g++ -O2 -std=c++17 sp.cpp -o sp
// Run:    ./sp data results 7.08
//         (data dir with bay_<N>.edges.csv / bay_<N>.coords.csv, output dir, heuristic scale)
//
// Outputs: <out>/results.csv and <out>/paths.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;

struct Edge { int to; long long w; };

struct Graph {
    int n = 0;
    vector<vector<Edge>> adj;   // adjacency list
    vector<double> lon, lat;    // degrees
};



struct Result {
    bool found = false;
    long long cost = -1;
    vector<int> path;
    long long explored = 0;     // vertices settled (popped, non-stale)
    long long edges = 0;        // outgoing edges examined from settled vertices
    double ms = 0;
};

Graph load_graph(const string& prefix) {
    Graph g;
    string line;
    ifstream fc(prefix + ".coords.csv");
    if (!fc) throw runtime_error("cannot open " + prefix + ".coords.csv");
    getline(fc, line);  // header
    while (getline(fc, line)) {
        istringstream ss(line);
        string id, lo, la;
        getline(ss, id, ',');
        getline(ss, lo, ',');
        getline(ss, la, ',');
        g.lon.push_back(stod(lo));
        g.lat.push_back(stod(la));
    }
    g.n = static_cast<int>(g.lon.size());
    g.adj.assign(g.n, {});
    ifstream fe(prefix + ".edges.csv");
    if (!fe) throw runtime_error("cannot open " + prefix + ".edges.csv");
    getline(fe, line);  // header
    while (getline(fe, line)) {
        istringstream ss(line);
        string u, v, w;
        getline(ss, u, ',');
        getline(ss, v, ',');
        getline(ss, w, ',');
        g.adj[stoi(u)].push_back({stoi(v), stoll(w)});
    }
    return g;
}

double haversine(const Graph& g, int a, int b) {
    const double R = 6371000.0, d2r = 3.14159265358979323846 / 180.0;
    double p1 = g.lat[a] * d2r, p2 = g.lat[b] * d2r;
    double dp = p2 - p1, dl = (g.lon[b] - g.lon[a]) * d2r;
    double s = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2);
    return 2 * R * asin(sqrt(s));
}

vector<int> build_path(const vector<int>& par, int t) {
    vector<int> p;
    for (int v = t; v != -1; v = par[v]) p.push_back(v);
    reverse(p.begin(), p.end());
    return p;
}

Result dijkstra(const Graph& g, int s, int t) {
    auto t0 = Clock::now();
    Result r;
    const long long INF = numeric_limits<long long>::max();
    vector<long long> dist(g.n, INF);
    vector<int> par(g.n, -1);
    vector<char> done(g.n, 0);
    using P = pair<long long, int>;
    priority_queue<P, vector<P>, greater<P>> pq;  // min-heap
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (done[u]) continue;  // stale entry
        done[u] = 1;
        r.explored++;
        if (u == t) { r.found = true; break; }
        for (const Edge& e : g.adj[u]) {
            r.edges++;
            if (d + e.w < dist[e.to]) {
                dist[e.to] = d + e.w;
                par[e.to] = u;
                pq.push({dist[e.to], e.to});
            }
        }
    }
    if (r.found) { r.cost = dist[t]; r.path = build_path(par, t); }
    r.ms = chrono::duration<double, milli>(Clock::now() - t0).count();
    return r;
}

Result astar(const Graph& g, int s, int t, double scale) {
    auto t0 = Clock::now();
    Result r;
    const long long INF = numeric_limits<long long>::max();
    vector<long long> gc(g.n, INF);
    vector<int> par(g.n, -1);
    vector<char> done(g.n, 0);
    using P = pair<double, int>;  // (f, vertex)
    priority_queue<P, vector<P>, greater<P>> pq;
    auto h = [&](int v) { return scale * haversine(g, v, t); };
    gc[s] = 0;
    pq.push({h(s), s});
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        if (done[u]) continue;  // consistent h => first pop is optimal
        done[u] = 1;
        r.explored++;
        if (u == t) { r.found = true; break; }
        for (const Edge& e : g.adj[u]) {
            r.edges++;
            if (done[e.to]) continue;
            long long ng = gc[u] + e.w;
            if (ng < gc[e.to]) {
                gc[e.to] = ng;
                par[e.to] = u;
                pq.push({(double)ng + h(e.to), e.to});
            }
        }
    }
    if (r.found) { r.cost = gc[t]; r.path = build_path(par, t); }
    r.ms = chrono::duration<double, milli>(Clock::now() - t0).count();
    return r;
}

int main(int argc, char** argv) {
    string data = argc > 1 ? argv[1] : "data";
    string out = argc > 2 ? argv[2] : "results";
    double scale = argc > 3 ? stod(argv[3]) : 7.08;
    const int PAIRS = 5;
    const unsigned SEED = 42;
    const vector<int> sizes = {1000, 5000, 10000, 25000, 50000};

    filesystem::create_directories(out);
    ofstream res(out + "/results.csv"), paths(out + "/paths.csv");
    res << "size,pair,src,dst,algo,found,cost,explored,edges,time_ms,path_len\n";
    paths << "size,pair,algo,path\n";

    for (int size : sizes) {
        Graph g = load_graph(data + "/bay_" + to_string(size));
        mt19937 rng(SEED + size);
        uniform_int_distribution<int> pick(0, g.n - 1);
        int done_pairs = 0, skipped = 0;
        while (done_pairs < PAIRS && skipped < 10000) {
            int s = pick(rng), t = pick(rng);
            if (s == t) continue;
            Result d = dijkstra(g, s, t);
            if (!d.found) { skipped++; continue; }  // unreachable: skip, but handled cleanly
            Result a = astar(g, s, t, scale);
            if (a.cost != d.cost)
                cerr << "WARNING: cost mismatch size=" << size << " pair=" << done_pairs
                     << " dijkstra=" << d.cost << " astar=" << a.cost << "\n";
            auto write = [&](const char* name, const Result& r) {
                res << size << "," << done_pairs << "," << s << "," << t << "," << name << ","
                    << r.found << "," << r.cost << "," << r.explored << "," << r.edges << ","
                    << r.ms << "," << r.path.size() << "\n";
                paths << size << "," << done_pairs << "," << name << ",\"";
                for (size_t i = 0; i < r.path.size(); i++) paths << (i ? " " : "") << r.path[i];
                paths << "\"\n";
            };
            write("dijkstra", d);
            write("astar", a);
            done_pairs++;
        }
        cout << "size " << size << ": " << done_pairs << " pairs done, "
             << skipped << " unreachable pairs skipped\n";
    }
    return 0;
}