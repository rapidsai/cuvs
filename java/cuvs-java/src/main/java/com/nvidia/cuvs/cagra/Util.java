/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.invoke.VarHandle;

public class Util {
	/**
	 * A utility method for getting an instance of {@link MemorySegment} for a {@link String}.
	 * 
	 * @param str the string for the expected {@link MemorySegment}
	 * @return an instance of {@link MemorySegment}
	 */
	protected static MemorySegment buildMemorySegment(Linker linker, Arena arena, String str) {
		StringBuilder sb = new StringBuilder(str).append('\0');
		MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(sb.length(),
				linker.canonicalLayouts().get("char"));
		MemorySegment stringMemorySegment = arena.allocate(stringMemoryLayout);

		for (int i = 0; i < sb.length(); i++) {
			VarHandle varHandle = stringMemoryLayout.varHandle(PathElement.sequenceElement(i));
			varHandle.set(stringMemorySegment, 0L, (byte) sb.charAt(i));
		}
		return stringMemorySegment;
	}

	/**
	 * A utility method for building a {@link MemorySegment} for a 2D float array.
	 * 
	 * @param data The 2D float array for which the {@link MemorySegment} is needed
	 * @return an instance of {@link MemorySegment}
	 */
	protected static MemorySegment buildMemorySegment(Linker linker, Arena arena, float[][] data) {
		long rows = data.length;
		long cols = data[0].length;

		MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows,
				MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
		MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				VarHandle element = dataMemoryLayout.arrayElementVarHandle(PathElement.sequenceElement(r),
						PathElement.sequenceElement(c));
				element.set(dataMemorySegment, 0, 0, data[r][c]);
			}
		}

		return dataMemorySegment;
	}


}
