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

func (m *CuvsPoolMemory) Close() error {
	m.ch <- CuvsMemoryRelease
	err := <-m.errCh
	close(m.ch)
	close(m.errCh)
	return err
}
