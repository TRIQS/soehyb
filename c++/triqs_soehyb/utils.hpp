/*******************************************************************************
 *
 * triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2025, H. U.R. Strand
 *
 * triqs_soehyb is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

// https://en.wikipedia.org/wiki/Exponentiation_by_squaring

template<typename T> static constexpr inline T pown(T x, unsigned p) {
    T result = 1;
    while (p) {
        if (p & 0x1) result *= x;
        x *= x;
        p >>= 1;
    }
    return result;
}
