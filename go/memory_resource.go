package cuvs

// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"
import (
	"runtime"
)

type CuvsMemoryCommand struct {
	cmd  int
	done chan struct{} // Response channel
}

const (
	CuvsMemoryNew = iota
	CuvsMemoryRelease
)

type CuvsPoolMemory struct {
	ch                        chan CuvsMemoryCommand
	initial_pool_size_percent int
	max_pool_size_percent     int
	managed                   bool
}

func NewCuvsPoolMemory(initial_pool_size_percent int, max_pool_size_percent int, managed bool) *CuvsPoolMemory {
	c := CuvsPoolMemory{ch: make(chan CuvsMemoryCommand), initial_pool_size_percent: initial_pool_size_percent, max_pool_size_percent: max_pool_size_percent, managed: managed}
	c.start()
	return &c
}

func (m *CuvsPoolMemory) start() {
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for command := range m.ch {
			switch command.cmd {
			case CuvsMemoryNew:
				CheckCuvs(CuvsError(C.cuvsRMMPoolMemoryResourceEnable(C.int(m.initial_pool_size_percent), C.int(m.max_pool_size_percent), C._Bool(m.managed))))
			case CuvsMemoryRelease:
				CheckCuvs(CuvsError(C.cuvsRMMMemoryResourceReset()))
			}

			close(command.done)
		}
	}()
}
func (m *CuvsPoolMemory) Close() {
	close(m.ch)
}
func (m *CuvsPoolMemory) Instantiate() {
	done := make(chan struct{})
	m.ch <- CuvsMemoryCommand{
		cmd:  CuvsMemoryNew,
		done: done,
	}
	<-done // Wait for completion
}

func (m *CuvsPoolMemory) Release() *CuvsPoolMemory {
	done := make(chan struct{})
	m.ch <- CuvsMemoryCommand{
		cmd:  CuvsMemoryRelease,
		done: done,
	}
	<-done // Wait for completion
	return m
}
