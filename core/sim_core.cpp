#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>

namespace py = pybind11;

static std::mt19937 g_rng(42);
static std::uniform_real_distribution<double> g_uniform(0.0, 1.0);
static std::normal_distribution<double> g_normal(0.0, 1.0);

float add(float a, float b) {
    return a + b;
}

void step_core(
    py::array_t<double> positions,
    py::array_t<bool> active_mask,
    double D,
    double dt
) {
    auto pos_buf = positions.request();
    auto mask_buf = active_mask.request();
    
    double* pos = (double*)pos_buf.ptr;
    bool* mask = (bool*)mask_buf.ptr;
    
    int n_particles = pos_buf.shape[0];
    double sigma = std::sqrt(2.0 * D * dt);
    
    std::normal_distribution<double> dist(0.0, sigma);
    
    for (int i = 0; i < n_particles; i++) {
        if (mask[i]) {
            for (int j = 0; j < 3; j++) {
                pos[i * 3 + j] += dist(g_rng);
            }
        }
    }
}

void apply_geometry(
    py::array_t<double> positions,
    py::array_t<bool> active_mask,
    py::array_t<double> bounds,
    py::array_t<int> wall_modes,
    py::array_t<double> permeabilities,
    py::array_t<int> n_attempts,       
    py::array_t<int> n_crossings,     
    py::array_t<int> n_absorbed       
) {
    auto pos_buf = positions.request();
    auto mask_buf = active_mask.request();
    auto bounds_buf = bounds.request();
    auto modes_buf = wall_modes.request();
    auto perm_buf = permeabilities.request();
    auto att_buf = n_attempts.request();
    auto cross_buf = n_crossings.request();
    auto abs_buf = n_absorbed.request();
    
    double* pos = (double*)pos_buf.ptr;
    bool* mask = (bool*)mask_buf.ptr;
    double* bnds = (double*)bounds_buf.ptr;
    int* modes = (int*)modes_buf.ptr;
    double* perms = (double*)perm_buf.ptr;
    int* att = (int*)att_buf.ptr;
    int* cross = (int*)cross_buf.ptr;
    int* abs_cnt = (int*)abs_buf.ptr;
    
    int n_particles = pos_buf.shape[0];
    
    
    int dims[] = {0, 0, 1, 1, 2, 2};
    int dirs[] = {-1, 1, -1, 1, -1, 1};  
    
    for (int face = 0; face < 6; face++) {
        int dim = dims[face];
        int dir = dirs[face];
        double wall_pos = (dir < 0) ? bnds[dim * 2] : bnds[dim * 2 + 1];
        int mode = modes[face];
        double perm = perms[face];
        
        for (int i = 0; i < n_particles; i++) {
            if (!mask[i]) continue;
            
            double coord = pos[i * 3 + dim];
            bool over_wall = (dir < 0) ? (coord < wall_pos) : (coord > wall_pos);
            
            if (!over_wall) continue;
            
            if (mode == 0) {  
                pos[i * 3 + dim] = 2.0 * wall_pos - coord;
            }
            else if (mode == 1) {  
                pos[i * 3 + dim] = wall_pos;
                mask[i] = false;
                abs_cnt[face]++;
            }
            else if (mode == 2) {  
                att[face]++;
                double roll = g_uniform(g_rng);
                if (roll >= perm) {  
                    pos[i * 3 + dim] = 2.0 * wall_pos - coord;
                } else {  
                    cross[face]++;
                }
            }
        }
    }
}

PYBIND11_MODULE(sim_core, m) {
    m.def("step_core", &step_core, "Core simulation step",
          py::arg("positions"), py::arg("active_mask"), py::arg("D"), py::arg("dt"));
    m.def("apply_geometry", &apply_geometry, "Apply boundary conditions",
          py::arg("positions"), py::arg("active_mask"), py::arg("bounds"),
          py::arg("wall_modes"), py::arg("permeabilities"),
          py::arg("n_attempts"), py::arg("n_crossings"), py::arg("n_absorbed"));
}