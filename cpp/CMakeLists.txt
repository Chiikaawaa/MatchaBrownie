cmake_minimum_required(VERSION 3.15)
project(sim_core)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(sim_core sim_core.cpp)
