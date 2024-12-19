package cuvs

// #include <cuvs/core/c_api.h>
import "C"

import (
	"runtime"
)

type CuvsMemoryCommand int

const (
	CuvsMemoryNew = iota
	CuvsMemoryRelease
)

type CuvsPoolMemory struct {
	ch                        chan CuvsMemoryCommand
	errCh                     chan error
	initial_pool_size_percent int
	max_pool_size_percent     int
	managed                   bool
}

// Creates new CuvsPoolMemory struct
// initial_pool_size_percent is the initial size of the pool in percent of total available device memory
// max_pool_size_percent is the maximum size of the pool in percent of total available device memory
// managed is whether to use CUDA managed memory
func NewCuvsPoolMemory(initial_pool_size_percent int, max_pool_size_percent int, managed bool) (*CuvsPoolMemory, error) {
	c := CuvsPoolMemory{
		ch:                        make(chan CuvsMemoryCommand),
		errCh:                     make(chan error),
		initial_pool_size_percent: initial_pool_size_percent,
		max_pool_size_percent:     max_pool_size_percent,
		managed:                   managed,
	}

	c.start()
	c.ch <- CuvsMemoryNew

	if err := <-c.errCh; err != nil {
		return nil, err
	}

	return &c, nil
}

// Enables pool memory
func (m *CuvsPoolMemory) start() {
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		for command := range m.ch {
			var err error
			switch command {
			case CuvsMemoryNew:
				err = CheckCuvs(CuvsError(C.cuvsRMMPoolMemoryResourceEnable(
					C.int(m.initial_pool_size_percent),
					C.int(m.max_pool_size_percent),
					C._Bool(m.managed))))
				m.errCh <- err

			case CuvsMemoryRelease:
				err = CheckCuvs(CuvsError(C.cuvsRMMMemoryResourceReset()))
				m.errCh <- err
			}
		}
	}()
}

// Disables pool memory
func (m *CuvsPoolMemory) Close() error {
	m.ch <- CuvsMemoryRelease
	err := <-m.errCh
	close(m.ch)
	close(m.errCh)
	return err
}

func Example() error {
	mem, err := NewCuvsPoolMemory(60, 100, false)
	if err != nil {
		return err
	}

	err = mem.Close()
	if err != nil {
		return err
	}

	return nil
}
