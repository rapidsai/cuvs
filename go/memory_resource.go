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
	"fmt"
	"runtime"
	"sync"
)

var (
	threadRoutine sync.Once
	threadChan    chan func() error
	threadDone    chan struct{}
	threadWG      sync.WaitGroup
)

func initThreadRoutine() {
	threadChan = make(chan func() error)
	threadDone = make(chan struct{})
	threadWG.Add(1)
	go func() {
		defer threadWG.Done()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for {
			select {
			case f := <-threadChan:
				err := f()
				if err != nil {
					fmt.Printf("Error in thread routine: %v\n", err)
				}
			case <-threadDone:
				return
			}

		}
	}()
}

func runOnSameThread(f func() error) error {
	threadRoutine.Do(initThreadRoutine)
	errChan := make(chan error)
	threadChan <- func() error {
		err := f()
		errChan <- err
		return err
	}
	return <-errChan
}

func ShutdownThreadRoutine() {
	threadRoutine.Do(func() {}) // Ensure the routine was started
	close(threadDone)
	threadWG.Wait()
}

func EnablePoolMemoryResource(initial_pool_size_percent int, max_pool_size_percent int, managed bool) error {
	return runOnSameThread(func() error {
		err := CheckCuvs(CuvsError(C.cuvsRMMPoolMemoryResourceEnable(C.int(initial_pool_size_percent), C.int(max_pool_size_percent), C._Bool(managed))))
		if err != nil {
			return fmt.Errorf("failed to enable pool memory resource: %v", err)
		}
		return nil
	})
}

func ResetMemoryResource() error {
	err := runOnSameThread(func() error {

		err := CheckCuvs(CuvsError(C.cuvsRMMMemoryResourceReset()))
		if err != nil {
			return fmt.Errorf("failed to reset memory resource: %v", err)
		}
		return nil
	})
	if err != nil {
		return err
	}

	ShutdownThreadRoutine()
	return nil
}
