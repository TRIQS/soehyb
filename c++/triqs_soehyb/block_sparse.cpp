#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/utils.hpp>
#include <h5/format.hpp>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/print.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <stdexcept>

using namespace nda;

/////////////// BlockDiagOpFun (BDOF) class ///////////////
BlockDiagOpFun::BlockDiagOpFun(
    std::vector<nda::array<dcomplex,3>> &blocks,
    nda::vector_const_view<int> zero_block_indices) : 
    blocks(blocks), num_block_cols(blocks.size()), zero_block_indices(zero_block_indices) {}


BlockDiagOpFun::BlockDiagOpFun(int r, 
    nda::vector_const_view<int> block_sizes) : 
    num_block_cols(block_sizes.size()) {
        
    std::vector<nda::array<dcomplex,3>> blocks(num_block_cols);
    zero_block_indices = nda::make_regular(-1*nda::ones<int>(num_block_cols));
    for (int i = 0; i < num_block_cols; i++) {
        blocks[i] = nda::zeros<dcomplex>(r, block_sizes[i], block_sizes[i]);
    }
    this->blocks = blocks;
}

void BlockDiagOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
    zero_block_indices = nda::zeros<int>(num_block_cols);
}

void BlockDiagOpFun::set_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = block;
    zero_block_indices(i) = 0;
}

const std::vector<nda::array<dcomplex,3>>& BlockDiagOpFun::get_blocks() const {
    return blocks;
}

nda::array_const_view<dcomplex,3> BlockDiagOpFun::get_block(int i) const {
    return blocks[i];
}

nda::vector<int> BlockDiagOpFun::get_block_sizes() const {
    nda::vector<int> block_sizes(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        block_sizes(i) = blocks[i].shape(1);
    }
    return block_sizes;
}

int BlockDiagOpFun::get_block_size(int i) const {
    return blocks[i].shape(1);
}

int BlockDiagOpFun::get_num_block_cols() const {
    return num_block_cols;
}

int BlockDiagOpFun::get_zero_block_index(int i) const {
    return zero_block_indices(i);
}

void BlockDiagOpFun::set_blocks_dlr_coeffs(imtime_ops &itops) {    
    for (int i = 0; i < num_block_cols; i++) {
        blocks_dlr_coeffs[i] = itops.vals2coefs(blocks[i]);
    }
}

const std::vector<nda::array<dcomplex,3>>& 
    BlockDiagOpFun::get_blocks_dlr_coeffs() const {

    return blocks_dlr_coeffs;
}

nda::array_const_view<dcomplex,3>
    BlockDiagOpFun::get_block_dlr_coeffs(int i) const {

    return blocks_dlr_coeffs[i];
}

int BlockDiagOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_cols; i++) {
        if (zero_block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}

void BlockDiagOpFun::add_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = nda::make_regular(blocks[i] + block); // TODO: does this work?
}

std::string BlockDiagOpFun::hdf5_format() { return "BlockDiagOpFun"; }

void h5_write(h5::group g, const std::string& subgroup_name, const BlockDiagOpFun& BDOF) {
    auto sg = g.create_group(subgroup_name);
    h5::write_hdf5_format(sg, BDOF);
    for (int i = 0; i < BDOF.num_block_cols; i++) {
        h5::write(sg, "block_" + std::to_string(i), BDOF.blocks[i]);
    }
    h5::write(sg, "zero_block_indices", BDOF.zero_block_indices);
}

/////////////// BlockOp (BO) class ///////////////

BlockOp::BlockOp(
    nda::vector<int> &block_indices, 
    std::vector<nda::array<dcomplex,2>> &blocks) : 
    block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOp::BlockOp(
    nda::vector_const_view<int> block_indices, nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices), num_block_cols(block_indices.size()) {

    std::vector<nda::array<dcomplex,2>> blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            blocks[i] = nda::zeros<dcomplex>(block_sizes(i,0), block_sizes(i,1));
        }
        else {
            blocks[i] = nda::zeros<dcomplex>(1, 1);
        }
    }
    this->blocks = blocks;
}

BlockOp& BlockOp::operator+=(const BlockOp &F) {
    // BlockOp addition-assignment operator
    // @param[in] F BlockOp
    // TODO: exception handling
    for (int i = 0; i < this->num_block_cols; i++) {
        if (F.get_block_index(i) != -1) {
            this->blocks[i] += F.blocks[i];
        }
    }
    return *this;
}

void BlockOp::set_block_indices(
    nda::vector<int> &block_indices) {

    this->block_indices = block_indices;
    num_block_cols = block_indices.size();
}

void BlockOp::set_block_index(int i, int block_index) {
    block_indices(i) = block_index;
}

void BlockOp::set_blocks(
    std::vector<nda::array<dcomplex,2>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
}

void BlockOp::set_block(int i, nda::array_const_view<dcomplex,2> block) {
    blocks[i] = block;
}

nda::vector_const_view<int> BlockOp::get_block_indices() const {
    return block_indices;
}

int BlockOp::get_block_index(int i) const { return block_indices(i); }

const std::vector<nda::array<dcomplex,2>>& BlockOp::get_blocks() const {
    return blocks;
}

nda::array_const_view<dcomplex,2> BlockOp::get_block(int i) const {
    if (block_indices(i) == -1) {
        auto arr = nda::zeros<dcomplex>(1,1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

int BlockOp::get_num_block_cols() const { return num_block_cols; }

nda::array<int,2> BlockOp::get_block_sizes() const {
    auto block_sizes = nda::zeros<int>(num_block_cols,2);
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            block_sizes(i,0) = blocks[i].shape(0);
            block_sizes(i,1) = blocks[i].shape(1);
        }
        else {
            block_sizes(i,0) = -1;
            block_sizes(i,1) = -1;
        }
    }
    return block_sizes;
};
        
nda::vector<int> BlockOp::get_block_size(int i) const {
    auto block_size = nda::zeros<int>(2);
    if (block_indices(i) != -1) {
        block_size(0) = blocks[i].shape(0);
        block_size(1) = blocks[i].shape(1);
    }
    else {
        block_size() = -1;
    }
    return block_size;
};

int BlockOp::get_block_size(int block_ind, int dim) const {
    if (block_indices(block_ind) != -1) {
        return blocks[block_ind].shape(dim);
    }
    else {
        return -1;
    }
}

/////////////// BlockOpFun (BOF) class ///////////////

BlockOpFun::BlockOpFun(
    nda::vector_const_view<int> block_indices, 
    std::vector<nda::array<dcomplex,3>> &blocks) : 
    block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOpFun::BlockOpFun(
    int r, 
    nda::vector_const_view<int> block_indices, 
    nda::array_const_view<int,2> block_sizes) :
    block_indices(block_indices), num_block_cols(block_indices.size()) {
    
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            blocks[i] = nda::zeros<dcomplex>(r, block_sizes(i,0), block_sizes(i,1));
        }
        else {
            blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
        }
    }
}

void BlockOpFun::set_block_indices(
    nda::vector<int> &block_indices) {

    this->block_indices = block_indices;
    num_block_cols = block_indices.size();
}

void BlockOpFun::set_block_index(int i, int block_index) {
    block_indices(i) = block_index;
}

void BlockOpFun::set_blocks(
    std::vector<nda::array<dcomplex,3>> &blocks) {

    this->blocks = blocks;
    num_block_cols = blocks.size();
}

void BlockOpFun::set_block(int i, nda::array_const_view<dcomplex,3> block) {
    blocks[i] = block;
}

nda::vector_const_view<int> BlockOpFun::get_block_indices() const {
    return block_indices;
}

int BlockOpFun::get_block_index(int i) const {
    return block_indices(i);
}

const std::vector<nda::array<dcomplex,3>>& BlockOpFun::get_blocks() const {
    return blocks;
}

nda::array_const_view<dcomplex,3> BlockOpFun::get_block(int i) const {
    if (block_indices(i) == -1) {
        auto arr = nda::zeros<dcomplex>(1, 1, 1);
        return arr;
    }
    else {
        return blocks[i];
    }
}

int BlockOpFun::get_num_block_cols() const {
    return num_block_cols;
}

nda::array<int,2> BlockOpFun::get_block_sizes() const {
    auto block_sizes = nda::zeros<int>(num_block_cols,2);
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            block_sizes(i,0) = blocks[i].shape(1);
            block_sizes(i,1) = blocks[i].shape(2);
        }
        else {
            block_sizes(i,0) = -1;
            block_sizes(i,1) = -1;
        }
    }
    return block_sizes;
}

nda::vector<int> BlockOpFun::get_block_size(int i) const {
    auto block_size = nda::zeros<int>(2);
    if (block_indices(i) != -1) {
        block_size(0) = blocks[i].shape(1);
        block_size(1) = blocks[i].shape(2);
    }
    else {
        block_size() = -1;
    }
    return block_size;
}

void BlockOpFun::set_blocks_dlr_coeffs(imtime_ops &itops) {
    for (int i = 0; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            blocks_dlr_coeffs[i] = itops.vals2coefs(blocks[i]);
        }
    }
}

const std::vector<nda::array<dcomplex,3>>& BlockOpFun::get_blocks_dlr_coeffs() {
    return blocks_dlr_coeffs;
}

nda::array_const_view<dcomplex,3> BlockOpFun::get_block_dlr_coeffs(int i) const {
    return blocks_dlr_coeffs[i];
}

int BlockOpFun::get_num_time_nodes() const {
    for (int i; i < num_block_cols; i++) {
        if (block_indices(i) != -1) {
            return blocks[i].shape(0);
        }
    }
    return 0; // BlockDiagOpFun is all zeros anyways
}

/////////////// BackboneSignature ///////////////

BackboneSignature::BackboneSignature(nda::array<int,2> topology, int n) : 
    topology(topology), m(topology.extent(0)), // e.g. 2CA, m = 3, topology has 3 lines 
    n(n) {
    
    prefactor = nda::zeros<int>(m-1, 3); 
    /*        |  sign  |  sign on K  | exponent on K |
     p   ---------------------------------------------
     o   l    |        |             |               |
     l   ---------------------------------------------
     e   l'   |        |             |               |
        ---------------------------------------------
     i   l''  |        |             |               |
     n   ---------------------------------------------
     d   ...  |        |             |               |
     e
     x
    */
    vertices = nda::zeros<int>(2*m, 4); 
    /*       | bar? | dagger? |    pole index      |  sign on K
         --------------------------------------------------------
         0   |      |         |                    |
     v   --------------------------------------------------------
     e   1   |      |         |                    |
     r   --------------------------------------------------------
     t   2   |      |         |                    |
     e   --------------------------------------------------------
     x   ... |      |         |                    |
     #   --------------------------------------------------------
         2*m |      |         |                    |
         --------------------------------------------------------
    */
    edges = nda::zeros<int>(2*m-1, m-1); // 2*m-1 = # edges; m-1: each pole index
    /*            pole index      
              | l | l' | l'' | ... 
         -------------------------
         0    |   |    |     |    
     e   -------------------------
     d   1    |   |    |     |    
     g   -------------------------
     e   2    |   |    |     |    
     #   -------------------------
         ...  |   |    |     |    
         -------------------------
         2*m-1|   |    |     |    
         -------------------------
    */
}

void BackboneSignature::set_directions(nda::vector_const_view<int> fb) {
    // @param[in] fb forward/backward line information

    this->fb = fb; 
    if (m != fb.size()) {
        throw std::invalid_argument("topology and fb must have same # of vertices");
    }
    for (int i = 0; i < m; i++) {
        if (topology(i,0) >= topology(i,1)) {
            throw std::invalid_argument("first row of topology must contain smaller-numbered vertices");
        }
    }
    if (topology(0,0) != 0) throw std::invalid_argument("topology(0,0) must be 0"); 

    // set operator flags for each vertex, depending on fb

    vertices(0, 0) = 0; // operator on vertex 0 has no bar
    vertices(topology(0,1), 0) = 0; // operator on vertex connected to 0 has no bar
    if (fb(0) == 1) {
        vertices(0, 1) = 0; // annihilation operator on vertex 0
        vertices(topology(0,1), 1) = 1; // creation operator on vertex connected to 0
    } else {
        vertices(0, 1) = 1; // creation operator on vertex 0
        vertices(topology(0,1), 1) = 0; // annihilation operator on vertex connected to 0
    }

    for (int i = 1; i < m; i++) {
        vertices(topology(i,0), 0) = 0; // operator on vertex i has no bar
        vertices(topology(i,1), 0) = 1; // operator on vertex connected to i has a bar

        if (fb(i) == 1) {
            vertices(topology(i,0), 1) = 0; // annihilation operator on vertex i
            vertices(topology(i,1), 1) = 1; // creation operator on vertex i
        } else {
            vertices(topology(i,0), 1) = 1; // creation operator on vertex i
            vertices(topology(i,1), 1) = 0; // annihilation operator on vertex i
        }
    }
}

void BackboneSignature::reset_directions() {
    fb = 0; 
    vertices(_,range(0,2)) = 0; 
}

void BackboneSignature::set_pole_inds(
    nda::vector_const_view<int> pole_inds, 
    nda::vector_const_view<double> dlr_rf) {
    // @param[in] poles DLR/AAA poles (e.g. l, l' indices)

    this->pole_inds = pole_inds;
    for (int i = 1; i < m; i++) {
        if (fb(i) == 1) {
            if (dlr_rf(pole_inds(i-1)) <= 0) {
                // step 4(a)
                // place K^-_l F_nu at the right vertex
                vertices(topology(i,0), 2) = i-1;
                vertices(topology(i,0), 3) = -1; 
                // place K^+_l F^bar^dag_nu_l at the left vertex
                vertices(topology(i,1), 2) = i-1;
                vertices(topology(i,1), 3) = 1;
                // divide by K^-_l(0)
                prefactor(i-1, 0) = 1; 
                prefactor(i-1, 1) = -1; 
                prefactor(i-1, 2) = 1;
            } else {
                // step 4(b)
                // no K's on vertices
                // place K^+_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = 1;
                // divide by (K^+_l(0))^(# edges between vertices - 1)
                prefactor(i-1, 0) = 1; 
                prefactor(i-1, 1) = 1;
                prefactor(i-1, 2) = topology(i,1) - topology(i,0) - 1; 
            }
        }
        else {
            if (dlr_rf(pole_inds(i-1)) >= 0) {
                // step 4(a)
                // place K^+_l F^dag_pi at the right vertex
                vertices(topology(i,0), 2) = i-1;
                vertices(topology(i,0), 3) = 1;
                // place K^-_l F^bar_pi_l at the left vertex
                vertices(topology(i,1), 2) = i-1;
                vertices(topology(i,1), 3) = -1;
                // divide by -K^-+l(0)
                prefactor(i-1, 0) = -1; 
                prefactor(i-1, 1) = 1; 
                prefactor(i-1, 2) = 1;
            } else {
                // step 4(b)
                // no K's on vertices
                // place K^-_l on each edge between the two vertices
                for (int j = topology(i,0); j < topology(i,1); j++) edges(j, i-1) = -1;
                // divide by -(K^-_l(0))^(# edges between vertices - 1)
                prefactor(i-1, 0) = -1; 
                prefactor(i-1, 1) = -1;
                prefactor(i-1, 2) = topology(i,1) - topology(i,0) - 1; 
            }
        }
    }
}

void BackboneSignature::reset_pole_inds() {
    pole_inds = 0; 
    vertices(_,range(2,4)) = 0; 
    edges = 0; 
    prefactor = 0; 
}

void BackboneSignature::set_states(nda::vector_const_view<int> states) {
    // @param[in] states orbital+spin indices (e.g. lambda, mu indices), going 
    // right to left, excluding the ones associated with the special vertex

    if (states.size() != n-2) {
        throw std::invalid_argument("states must have size n - 2");
    }
}

int BackboneSignature::get_prefactor(int pole_ind, int i) {return prefactor(pole_ind, i);}

int BackboneSignature::get_vertex(int num, int i) {return vertices(num, i);}

int BackboneSignature::get_edge(int num, int pole_ind) {return edges(num, pole_ind);}

int BackboneSignature::get_topology(int i, int j) {return topology(i, j);}

int BackboneSignature::get_pole_ind(int i) {return pole_inds(i);}

/////////////// Utilities and operator overrides ///////////////

std::ostream& operator<<(std::ostream& os, BlockDiagOpFun &D) {
    // Print BlockDiagOpFun
    // @param[in] os output stream
    // @param[in] D BlockDiagOpFun
    // @return output stream

    for (int i = 0; i < D.get_num_block_cols(); i++) {
        os << "Block " << i << ":\n" << D.get_block(i) << "\n";
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, BlockOp &F) {
    // Print BlockOp
    // @param[in] os output stream
    // @param[in] F BlockOp
    // @return output stream

    os << "Block indices: " << F.get_block_indices() << "\n";
    for (int i = 0; i < F.get_num_block_cols(); i++) {
        if (F.get_block_indices()[i] == -1) {
            os << "Block " << i << ": 0\n";
        }
        else {
            os << "Block " << i << ":\n" << F.get_block(i) << "\n";
        }
    }
    return os;
};

BlockOp dagger_bs(BlockOp const &F) {
    // Evaluate F^dagger in block-sparse storage
    // @param[in] F F operator
    // @return F^dagger operator

    int num_block_cols = F.get_num_block_cols();
    int i, j;

    // find block indices for F^dagger
    nda::vector<int> block_indices_dag(num_block_cols);
    // initialize indices with -1
    block_indices_dag = -1;
    std::vector<nda::array<dcomplex,2>> blocks_dag(num_block_cols);
    for (i = 0; i < num_block_cols; ++i) {
        j = F.get_block_indices()[i];
        if (j != -1) {
            block_indices_dag[j] = i;
            blocks_dag[j] = nda::transpose(F.get_blocks()[i]);
        }
    }
    BlockOp F_dag(block_indices_dag, blocks_dag);
    return F_dag;
}

BlockOp operator*(const dcomplex c, const BlockOp &F) {    
    // Compute a product between a scalar and an BlockOp
    // @param[in] c dcomplex
    // @param[in] F BlockOp

    auto product = F;
    for (int i = 0; i < F.get_num_block_cols(); i++) {
        if (F.get_block_index(i) != -1) {
            auto prod_block = nda::make_regular(c*F.get_block(i));
            product.set_block(i, prod_block);
        }
    }
    return product;
}

std::ostream& operator<<(std::ostream& os, BackboneSignature &B) {
    // prefactor --> p_str
    std::string p_str = "1 / ("; 
    int sign = 1;
    for (int i = 0; i < B.m-1; i++) sign *= B.get_prefactor(i, 0); 
    if (sign == -1) p_str = "-" + p_str; 
    for (int i = 0; i < B.m-1; i++) {
        if (B.get_prefactor(i, 2) >= 1) {
            p_str += "K_{l";
            for (int j = 0; j < i; j++) p_str += "`";
            p_str += "}";
            if (B.get_prefactor(i, 1) == 1) p_str += "^+";
            else p_str += "^-"; 
        }
        if (B.get_prefactor(i, 2) > 1) {
            p_str += "^" + std::to_string(B.get_prefactor(i, 2)); 
        }
        p_str += "(0)";
        if (i < B.m-2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent = 30; 
    std::string diag_str = ""; 
    for (int i = 0; i < 30; i++) diag_str += " "; 
    diag_str += "0\n"; 
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2*B.m; i++) {
        // K factor
        if (B.get_vertex(i, 3) != 0) {
            v_str_tmp += "K_{l";
            for (int j = 0; j < B.get_vertex(i, 2); j++) v_str_tmp += "`";
            v_str_tmp += "}";
            if (B.get_vertex(i, 3) == 1) v_str_tmp += "^+";
            else v_str_tmp += "^-"; 
        }

        // F operator
        v_str_tmp += "F";
        if (B.get_vertex(i, 0) == 1 || B.get_vertex(i, 1) == 1) v_str_tmp += "^{"; 
        if (B.get_vertex(i, 0) == 1) v_str_tmp += "bar";
        if (B.get_vertex(i, 1) == 1) v_str_tmp += "dag";
        if (B.get_vertex(i, 0) == 1 || B.get_vertex(i, 1) == 1) v_str_tmp += "}"; 
        if (B.get_vertex(i, 0) != 1) v_str_tmp += "_" + std::to_string(i); 
        else {
            for (int j = 0; j < B.m; j++) if (B.get_topology(j,1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j,0));
        }
        // hybridization
        if (i == B.get_topology(0, 1)) {
            v_str_tmp += " Delta_{"; 
            if (B.get_vertex(i, 1) == 1) v_str_tmp += "0," + std::to_string(i) + "} "; 
            else v_str_tmp += std::to_string(i) + ",0}";
        } else v_str_tmp += " ";
        int vlen0 = v_str_tmp.size(); 
        for (int j = vlen0; j < 28; j++) v_str_tmp = " " + v_str_tmp; 
        diag_str += v_str_tmp + "--| \n"; 
        v_str_tmp = ""; 

        // edges --> e_str
        if (i < 2*B.m-1) {
            for (int j = 0; j < B.m-1; j++) {
                if (B.get_edge(i, j) != 0) {
                    e_str_tmp += "K_{l";
                    for (int k = 0; k < j; k++) e_str_tmp += "`";
                    e_str_tmp += "}";
                    if (B.get_edge(i, j) == 1) e_str_tmp += "^+ "; 
                    else e_str_tmp += "^- "; 
                }
            }
            e_str_tmp += "G ";
            for (int j = 0; j < 30; j++) diag_str += " ";
            diag_str += "| " + e_str_tmp + "\n"; 
            e_str_tmp = ""; 
        }
    }
    // diag_str += "\"; 
    for (int i = 0; i < 29; i++) diag_str += " "; 
    diag_str += "tau"; 

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
}

BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A) {
    // Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
    // @param[in] A BlockOpFun
    // @return BlockDiagOpFun

    int num_block_cols = A.get_num_block_cols();
    auto diag_blocks = A.get_blocks();
    auto block_indices = A.get_block_indices();
    auto zero_block_indices = nda::zeros<int>(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        int block_index = A.get_block_index(i);
        if (block_index == -1) {
            diag_blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
            zero_block_indices(i) = -1;
        }
        else if (block_index != i) {
            throw std::invalid_argument("BOF is not diagonal");
        }
    }

    return BlockDiagOpFun(diag_blocks, zero_block_indices);
}

BlockDiagOpFun NCA_bs(
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    BlockDiagOpFun const &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate NCA using block-sparse storage
    // @param[in] hyb hybridization function
    // @param[in] hyb_refl hybridization function eval'd at negative imag. times
    // @param[in] Gt Greens function
    // @param[in] F_list F operators
    // @return NCA term of self-energy
    
    // get F^dagger operators
    int num_Fs = Fs.size();
    auto F_dags = Fs;
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }
    // initialize self-energy, with same shape as Gt
    int r = Gt.get_num_time_nodes();
    BlockDiagOpFun Sigma(r, Gt.get_block_sizes());

    for (int fb = 0; fb <= 1; fb++) {
        // fb = 1 for forward line, 0 for backward line
        auto const &F1list = (fb) ? Fs : F_dags;
        auto const &F2list = (fb) ? F_dags : Fs;
        int sfM = -1;//(fb) ? -1 : 1; 
        
        for (int lam = 0; lam < num_Fs; lam++) {
            for (int kap = 0; kap < num_Fs; kap++) {
                auto &F1 = F1list[kap];
                auto &F2 = F2list[lam];
                int ind_path = 0;
                bool path_all_nonzero = true; // if set to false during backwards pass,
                // the i-th block of Sigma is zero, so no computation needed

                for (int i = 0; i < Gt.get_num_block_cols(); i++) {
                    // "backwards pass"
                    // for each self-energy block, find contributing blocks of factors
                    path_all_nonzero = true; 
                    ind_path = F1.get_block_index(i); // Sigma = F2 G F1
                    // ind_path = block-column index of F1 corresponding with
                    // block that contributes to i-th block of Sigma
                    if (ind_path == -1 
                        || Gt.get_zero_block_index(ind_path) == -1 
                        || F2.get_block_index(ind_path) == -1) {
                        path_all_nonzero = false; // one of the blocks of F1,
                        // Gt, Ft that contribute to block i of Sigma is zero
                    }

                    // matmuls
                    // if path involves all nonzero blocks, compute product
                    // of blocks indexed by ind_path
                    if (path_all_nonzero) {
                        auto block = nda::zeros<dcomplex>(r, Gt.get_block_size(i), Gt.get_block_size(i));
                        for (int t = 0; t < r; t++) {
                            if (fb == 1) {
                                block(t,_,_) = hyb(t, lam, kap) * nda::matmul(
                                    F2.get_block(ind_path), 
                                    nda::matmul(
                                        Gt.get_block(ind_path)(t,_,_), 
                                        F1.get_block(i)));
                            } else {
                                block(t,_,_) = hyb_refl(t, kap, lam) * nda::matmul(
                                    F2.get_block(ind_path), 
                                    nda::matmul(
                                        Gt.get_block(ind_path)(t,_,_), 
                                        F1.get_block(i)));
                            }
                        }
                        block = sfM * block;
                        Sigma.add_block(i, block);
                    }
                }
            }
        }
    }
    
    return Sigma;
}

nda::array<dcomplex,3> NCA_dense(
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,3> Fs,
    nda::array_const_view<dcomplex,3> F_dags) {

    // initialize self-energy, with same shape as Gt
    int r = Gt.extent(0);
    int N = Gt.extent(1);
    nda::array<dcomplex,3> Sigma(r, N, N);
    int n = Fs.extent(0);

    for (int fb = 0; fb <= 1; fb++) {
        // fb = 1 for forward line, 0 for backward line
        auto const &F1list = (fb) ? Fs : F_dags;
        auto const &F2list = (fb) ? F_dags : Fs;
        int sfM = -1;//(fb) ? -1 : 1; 
        
        for (int lam = 0; lam < n; lam++) {
            for (int kap = 0; kap < n; kap++) {
                auto F1 = F1list(kap,_,_);
                auto F2 = F2list(lam,_,_);

                for (int t = 0; t < r; t++) {
                    if (fb == 1) {
                        Sigma(t,_,_) += sfM * hyb(t, lam, kap) * nda::matmul(
                            F2, 
                            nda::matmul(
                                Gt(t,_,_), 
                                F1));
                    } else {
                        Sigma(t,_,_) += sfM * hyb_refl(t, lam, kap) * nda::matmul(
                            F2, 
                            nda::matmul(
                                Gt(t,_,_), 
                                F1));
                    }
                }
            }
        }
    }
    
    return Sigma;
}

nda::array<double,2> K_mat(
    nda::vector_const_view<double> dlr_it,
    nda::vector_const_view<double> dlr_rf,
    double beta = 1.0) {
    // @brief Build matrix of evaluations of K at imag times and real freqs
    // @param[in] dlr_it DLR imaginary time nodes
    // @param[in] dlr_rf DLR real frequencies
    // @return matrix of K evalutions

    int r = dlr_it.shape(0); // number of times = number of freqs
    nda::array<double,2> K(r, r);
    for (int k = 0; k < r; k++) {
        for (int l = 0; l < r; l++) {
            K(k,l) = k_it(dlr_it(k), dlr_rf(l), beta);
        }
    }
    return K;
}

nda::array<dcomplex,3> convolve_rectangular(
    imtime_ops &itops, 
    double beta, 
    nda::array<dcomplex,3> f, 
    nda::array<dcomplex,3> g) {

    nda::array<dcomplex,3> h(f.extent(0), f.extent(1), g.extent(2));
    if (f.extent(2) != g.extent(1)) {
        std::cout << "# cols f = " << f.extent(2) << std::endl;
        std::cout << "# rows g = " << g.extent(1) << std::endl;
        throw std::invalid_argument("incompatible matrices");
    } else if (f.extent(0) != g.extent(0)) {
        throw std::invalid_argument("time indices do not match");
    }

    for (int i = 0; i < f.extent(1); i++) {
        for (int j = 0; j < g.extent(2); j++) {
            for (int k = 0; k < f.extent(2); k++) {
                h(_,i,j) += itops.convolve(beta, 
                    Fermion, 
                    itops.vals2coefs(f(_,i,k)), 
                    itops.vals2coefs(g(_,k,j)), 
                    TIME_ORDERED);
            }
        }
    }

    return h;
}

BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double,2>> H_blocks, 
    nda::vector<int> H_block_inds, 
    double beta, 
    nda::vector_const_view<double> dlr_it) {

    int num_block_cols = H_block_inds.size();
    nda::vector<int> H_block_sizes(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        H_block_sizes(i) = H_blocks[i].extent(0);
    }
    
    int r = dlr_it.size();
    
    double tr_exp_minusbetaH = 0;
    std::vector<nda::array<double,1>> H_evals(num_block_cols);
    std::vector<nda::array<double,2>> H_evecs(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
        if (H_block_inds(i) != -1) {
            if (H_block_sizes(i) == 1) {
                H_evals[i] = nda::array<double,1>{H_blocks[i](0,0)};
                H_evecs[i] = nda::array<double,2>{{1}};
            } else {
                auto H_block_eig = nda::linalg::eigenelements(H_blocks[i]);
                H_evals[i] = std::get<0>(H_block_eig);
                H_evecs[i] = std::get<1>(H_block_eig);
            }
            tr_exp_minusbetaH += nda::sum(exp(-beta*H_evals[i]));
        }
        else {
            H_evals[i] = nda::zeros<double>(H_block_sizes(i));
            H_evecs[i] = nda::eye<double>(H_block_sizes(i));
            tr_exp_minusbetaH += 1.0*H_block_sizes(i); // 0 entry in the diagonal
        }
    }

    auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;

    // TODO finish writing this function
    // create test combining call to this with beginning of twoband
    // start dedicated two_band test
    // check that Gt and H have the same zero block indices

    auto Gt = BlockDiagOpFun(r, H_block_sizes);
    for (int i = 0; i < num_block_cols; i++) {
        auto Gt_block = nda::array<dcomplex,3>(r, H_block_sizes(i), H_block_sizes(i));
        auto Gt_temp = nda::make_regular(0*H_blocks[i]);
        for (int t = 0; t < r; t++) {
            for (int j = 0; j < H_block_sizes(i); j++) {
                Gt_temp(j,j) = -exp(-beta*dlr_it(t)*(H_evals[i](j) + eta_0));
            }
            Gt_block(t,_,_) = nda::matmul(
                H_evecs[i], 
                nda::matmul(Gt_temp, nda::transpose(H_evecs[i])));
        }
        Gt.set_block(i, Gt_block);
    }

    return Gt;
}

void OCA_bs_right_in_place(
    double beta, 
    imtime_ops &itops, 
    nda::vector_const_view<double> dlr_it, 
    double omega_l, 
    bool forward, 
    nda::array_const_view<dcomplex,3> Gt0, 
    nda::array_const_view<dcomplex,3> Gt1, 
    nda::array_const_view<dcomplex,2> Flam, 
    nda::array_view<dcomplex,3> T) {

    int r = Gt0.extent(0);
    // nda::array<dcomplex,3> T(r,Flam.extent(0),Flam.extent(1));

    if (forward) {
        if (omega_l <= 0) {
            // 1. multiply F_lambda G(tau_1) K^-(tau_1)
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), -omega_l) * nda::matmul(Flam,Gt0(t,_,_));
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt1), itops.vals2coefs(T), TIME_ORDERED);
        }
        else {
            // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), omega_l) * nda::matmul(Gt1(t,_,_),Flam);
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(T), itops.vals2coefs(Gt0), TIME_ORDERED);
        }
    } else {
        if (omega_l >= 0) {
            // 1. multiply F_lambda G(tau_1) K^+(tau_1)
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), omega_l) * nda::matmul(Flam,Gt0(t,_,_));
            } 
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt1), itops.vals2coefs(T), TIME_ORDERED);
        } else {
            // 1. multiply G(tau_2-tau_1) K^-(tau_2-tau_1) F_lambda
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), -omega_l) * nda::matmul(Gt1(t,_,_),Flam);
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(T), itops.vals2coefs(Gt0), TIME_ORDERED); 
        }
    }
}

void OCA_bs_middle_in_place(
    bool forward, 
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    nda::array_const_view<dcomplex,3> Fkaps, 
    nda::array_const_view<dcomplex,3> Fmus, 
    nda::array_view<dcomplex,3> Tin, 
    nda::array_view<dcomplex,3> Tout, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu) {
    
    int num_Fs = Fkaps.extent(0);
    int r = hyb.extent(0);
    // 3. for each kappa, multiply by F_kappa from right
    // auto Tkap = nda::zeros<dcomplex>(num_Fs, r, T.extent(1), Fkaps.extent(2));
    for (int kap = 0; kap < num_Fs; kap++) {
        for (int t = 0; t < r; t++) {
            Tkaps(kap,t,_,_) = nda::matmul(Tin(t,_,_),Fkaps(kap,_,_));
        }
    }

    // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
    // nda::array<dcomplex,3> Tmu(r, T.extent(1), Fkaps.extent(2));
    Tout = 0;
    auto U = nda::zeros<dcomplex>(r, Fmus.extent(1), Fkaps.extent(2));
    for (int mu = 0; mu < num_Fs; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < num_Fs; kap++) {
            for (int t = 0; t < r; t++) {
                if (forward) {
                    Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
                } else {
                    Tmu(t,_,_) += hyb_refl(t,kap,mu)*Tkaps(kap,t,_,_);
                }
            }
        }
        // 5. multiply by F^dag_mu and sum over mu
        for (int t = 0; t < r; t++) {
            auto Fmu = Fmus(mu,_,_);
            Tout(t,_,_) += nda::matmul(Fmu, Tmu(t,_,_));
        }
    }
}

void OCA_bs_left_in_place(
    double beta, 
    imtime_ops &itops, 
    nda::vector_const_view<double> dlr_it, 
    double omega_l, 
    bool forward, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,2> Fbar, 
    nda::array_view<dcomplex,3> Tin, 
    nda::array_view<dcomplex,3> Tout, 
    nda::array_view<dcomplex,3> GKt) {

    int r = Gt.extent(0);
    // nda::array<dcomplex,3> Sigma_l(r,Fbar.extent(0),U.extent(2));
    // auto dlr_it = itops.get_itnodes();

    if (forward) {
        if (omega_l <= 0) {
            // 6. convolve by G
            Tin = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(Tin), TIME_ORDERED);
            // 7. mTinltiply by Fbar
            for (int t = 0; t < r; t++) {
                Tout(t,_,_) = nda::matmul(Fbar, Tin(t,_,_));
            }
        }
        else {
            // 6. convolve by G K^+
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t), omega_l)*Gt(t,_,_);
            }
            Tin = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(Tin), TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                Tout(t,_,_) = nda::matmul(Fbar, Tin(t,_,_));
            }
        }
    } else {
        if (omega_l >= 0) {
            // 6. convolve by G
            Tin = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(Tin), TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                Tout(t,_,_) = nda::matmul(Fbar, Tin(t,_,_));
            }
        }
        else {
            // 6. convolve by G K^-
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t),-omega_l)*Gt(t,_,_);
            }
            Tin = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(Tin), TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                Tout(t,_,_) = nda::matmul(Fbar, Tin(t,_,_));
            }
        }
    }
}

BlockDiagOpFun OCA_bs(
    nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    const BlockDiagOpFun &Gt, 
    const std::vector<BlockOp> &Fs) {
    // Evaluate OCA using block-sparse storage
    // @param[in] hyb hybridization on imaginary-time grid
    // @param[in] itops cppdlr imaginary time object
    // @param[in] beta inverse temperature
    // @param[in] Gt Greens function at times dlr_it with DLR coefficients
    // @param[in] Fs F operators
    // @return OCA term of self-energy

    // TODO: exceptions for bad argument sizes

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.shape(0);

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl); 

    // get F^dagger operators
    int num_Fs = Fs.size();
    auto F_dags = Fs;
    for (int i = 0; i < num_Fs; ++i) {
        F_dags[i] = dagger_bs(Fs[i]);
    }

    // compute Fbars and Fdagbars
    auto Fbar_indices = Fs[0].get_block_indices();
    auto Fbar_sizes = Fs[0].get_block_sizes();
    auto Fdagbar_indices = F_dags[0].get_block_indices();
    auto Fdagbar_sizes = F_dags[0].get_block_sizes();
    std::vector<std::vector<BlockOp>> Fdagbars(
        num_Fs, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fdagbar_indices, Fdagbar_sizes)));
    std::vector<std::vector<BlockOp>> Fbarsrefl(
        num_Fs, 
        std::vector<BlockOp>(
            r, 
            BlockOp(Fbar_indices, Fbar_sizes)));
    for (int lam = 0; lam < num_Fs; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < num_Fs; nu++) {
                Fdagbars[lam][l] += hyb_coeffs(l,nu,lam)*F_dags[nu];
                Fbarsrefl[lam][l] += hyb_refl_coeffs(l,lam,nu)*Fs[nu];
            }
        }
    }

    // initialize self-energy
    BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
    int num_block_cols = Gt.get_num_block_cols();

    // preallocation
    bool path_all_nonzero;
    nda::vector<int> ind_path(3);
    nda::vector<int> block_dims(5); // intermediate block dimensions

    // loop over hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
        for (int fb2 = 0; fb2 <= 1; fb2++) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1) ? Fs : F_dags;
            auto const &F2list = (fb2) ? Fs : F_dags;
            auto const &F3list = (fb1) ? F_dags : Fs;
            auto const Fbar_array = (fb2) ? Fdagbars : Fbarsrefl;
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            for (int i = 0; i < num_block_cols; i++) {

                // "backwards pass"
                // 
                // for each self-energy block, find contributing blocks of factors
                // 
                // paths_all_nonzero: false if for i-th block of 
                // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
                // 
                // ind_path: a vector of column indices of the 
                // blocks of the factors that contribute. if paths_all_nonzero is 
                // false at this index, values in ind_path are garbage.
                //
                // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
                // i.e, index path is independent of kappa, mu
                path_all_nonzero = true;
                auto Sigma_block_i = nda::make_regular(0*Sigma.get_block(i));

                ind_path(0) = F1list[0].get_block_index(i);
                if (ind_path(0) == -1 || Gt.get_zero_block_index(ind_path(0)) == -1) {
                    path_all_nonzero = false;
                }
                else {
                    block_dims(0) = F1list[0].get_block_size(i,1);
                    block_dims(1) = F1list[0].get_block_size(i,0);
                    ind_path(1) = F2list[0].get_block_index(ind_path(0));
                    if (ind_path(1) == -1 || Gt.get_zero_block_index(ind_path(1)) == -1) {
                        path_all_nonzero = false;
                    }
                    else {
                        block_dims(2) = F2list[0].get_block_size(ind_path(0),0);
                        ind_path(2) = F3list[0].get_block_index(ind_path(1));
                        if (ind_path(2) == -1 
                            || Gt.get_zero_block_index(ind_path(2)) == -1 
                            || Fbar_array[0][0].get_block_index(ind_path(2)) == -1) {
                            path_all_nonzero = false;
                        }
                        else {
                            block_dims(3) = F3list[0].get_block_size(ind_path(1),0);
                            block_dims(4) = Fbar_array[0][0].get_block_size(ind_path(2),0);
                        }
                    }
                }

                // matmuls and convolutions
                if (path_all_nonzero) {
                    // preallocate for i-th block
                    auto Sigma_l = nda::make_regular(0*Sigma.get_block(i));
                    // sizes of intermediate matrices are known
                    nda::array<dcomplex,3> Tright(r, block_dims(2), block_dims(1)); // output of OCA_bs_right
                    nda::array<dcomplex,3> Tmid(r, block_dims(3), block_dims(0)); // output of OCA_bs_middle
                    nda::array<dcomplex,4> Tkaps(num_Fs, r, block_dims(2), block_dims(0)); // storage in OCA_bs_middle
                    nda::array<dcomplex,3> Tmu(r, block_dims(2), block_dims(0)); // storage in OCA_bs_middle
                    nda::array<dcomplex,3> Tleft(r, block_dims(4), block_dims(0)); // output of OCA_bs_left
                    nda::array<dcomplex,3> GKt(r, block_dims(3), block_dims(3)); // storage in OCA_bs_left

                    // TODO: make Fs have blocks that are 3D nda::array?
                    auto Fkaps = nda::zeros<dcomplex>(
                        num_Fs, 
                        F1list[0].get_block_size(i,0), 
                        F1list[0].get_block_size(i,1));
                    auto Fmus = nda::zeros<dcomplex>(
                        num_Fs, 
                        F3list[0].get_block_size(ind_path(1),0),
                        F3list[0].get_block_size(ind_path(1),1));
                    for (int j = 0; j < num_Fs; j++) {
                        Fkaps(j,_,_) = F1list[j].get_block(i);
                        Fmus(j,_,_) = F3list[j].get_block(ind_path(1));
                    }

                    for (int l = 0; l < r; l++) {
                        Sigma_l = 0;
                        for (int lam = 0; lam < num_Fs; lam++) {
                            OCA_bs_right_in_place(
                                beta, itops, dlr_it, dlr_rf(l), (fb2==1), 
                                Gt.get_block(ind_path(0)), 
                                Gt.get_block(ind_path(1)), 
                                F2list[lam].get_block(ind_path(0)), 
                                Tright);
                            OCA_bs_middle_in_place((fb1==1), 
                                hyb, 
                                hyb_refl, 
                                Fkaps, 
                                Fmus, 
                                Tright, Tmid, Tkaps, Tmu);
                            OCA_bs_left_in_place(
                                beta, itops, dlr_it, dlr_rf(l), (fb2==1), 
                                Gt.get_block(ind_path(2)), 
                                Fbar_array[lam][l].get_block(ind_path(2)), 
                                Tmid, Tleft, GKt);
                            Sigma_l += Tleft;
                        } // sum over lambda

                        // prefactor with Ks
                        if (fb2 == 1) {
                            if (dlr_rf(l) <= 0) {
                                for (int t = 0; t < r; t++) {
                                    Sigma_l(t,_,_) = k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t,_,_);
                                }
                                Sigma_l = Sigma_l/k_it(0, -dlr_rf(l));
                            } else {
                                Sigma_l = Sigma_l/k_it(0, dlr_rf(l));
                            }
                        } else {
                            if (dlr_rf(l) >= 0) {
                                for (int t = 0; t < r; t++) {
                                    Sigma_l(t,_,_) = k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t,_,_);
                                }
                                Sigma_l = Sigma_l/k_it(0, dlr_rf(l));
                            } else {
                                Sigma_l = Sigma_l/k_it(0, -dlr_rf(l));
                            }
                        }
                        Sigma_block_i += nda::make_regular(sfM*Sigma_l);
                    } // sum over l
                    Sigma.add_block(i, Sigma_block_i);
                } // end if(path_all_nonzero)
            } // end loop over i
        } // end loop over fb2
    } // end loop over fb1
    return Sigma;
}

void OCA_dense_right_in_place(
    double beta, 
    imtime_ops &itops, 
    nda::vector_const_view<double> dlr_it, 
    double omega_l, 
    bool forward, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,2> Flam, 
    nda::array_view<dcomplex,3> T) {
    
    int r = Gt.extent(0);
    int N = Gt.extent(1);

    if (forward) {
        if (omega_l <= 0) {
            // 1. multiply F_lambda G(tau_1) K^-(tau_1)
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), -omega_l) * nda::matmul(Flam,Gt(t,_,_));
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
            // T = itops.convolve(Gt_conv, T); 
        }
        else {
            // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), omega_l) * nda::matmul(Gt(t,_,_),Flam);
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
        }
    } else {
        if (omega_l >= 0) {
            // 1. multiply F_lambda G(tau_1) K^+(tau_1)
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), omega_l) * nda::matmul(Flam,Gt(t,_,_));
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
            // T = itops.convolve(Gt_conv, T); 
        } else {
            // 1. multiply G(tau_2-tau_1) K^-(tau_2-tau_1) F_lambda
            for (int t = 0; t < r; t++) {
                T(t,_,_) = k_it(dlr_it(t), -omega_l) * nda::matmul(Gt(t,_,_),Flam);
            }
            // 2. convolve by G
            T = itops.convolve(beta, Fermion, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
        }
    }
}

void OCA_dense_middle_in_place(
    bool forward, 
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> hyb_refl, 
    nda::array_const_view<dcomplex,3> Fkaps, 
    nda::array_const_view<dcomplex,3> Fmus, 
    nda::array_view<dcomplex,3> T, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu
) {
    int num_Fs = Fkaps.extent(0);
    int r = hyb.extent(0);
    // 3. for each kappa, multiply by F_kappa from right
    for (int kap = 0; kap < num_Fs; kap++) {
        for (int t = 0; t < r; t++) {
            Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_),Fkaps(kap,_,_));
        }
    }

    // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
    for (int mu = 0; mu < num_Fs; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < num_Fs; kap++) {
            for (int t = 0; t < r; t++) {
                if (forward) {
                    Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
                } else {
                    Tmu(t,_,_) += hyb_refl(t,mu,kap)*Tkaps(kap,t,_,_);
                }
            }
        }
        // 5. multiply by F^dag_mu and sum over mu
        for (int t = 0; t < r; t++) {
            T(t,_,_) += nda::matmul(Fmus(mu,_,_), Tmu(t,_,_));
        }
    }
}

void OCA_dense_left_in_place(
    double beta, 
    imtime_ops &itops, 
    nda::vector_const_view<double> dlr_it, 
    double omega_l, 
    bool forward, 
    nda::array_const_view<dcomplex,3> Gt, 
    nda::array_const_view<dcomplex,2> Fbar, 
    nda::array_view<dcomplex,3> T, 
    nda::array_view<dcomplex,3> GKt) {

    int r = Gt.extent(0);
    int N = Gt.extent(1);

    if (forward) {
        if (omega_l <= 0) {
            // 6. convolve by G
            T = itops.convolve(
                    beta, 
                    Fermion, 
                    itops.vals2coefs(Gt), 
                    itops.vals2coefs(T), 
                    TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                T(t,_,_) = nda::matmul(Fbar, T(t,_,_));
            }
        }
        else {
            // 6. convolve by G K^+
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t), omega_l)*Gt(t,_,_);
            }
            T = itops.convolve(
                    beta, 
                    Fermion,
                    itops.vals2coefs(GKt), 
                    itops.vals2coefs(T),
                    TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                T(t,_,_) = nda::matmul(Fbar, T(t,_,_));
            }
        }
    } else {
        if (omega_l >= 0) {
            // 6. convolve by G
            T = itops.convolve(
                    beta, 
                    Fermion, 
                    itops.vals2coefs(Gt), 
                    itops.vals2coefs(T), 
                    TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                T(t,_,_) = nda::matmul(Fbar, T(t,_,_));
            }
        }
        else {
            // 6. convolve by G K^-
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t),-omega_l)*Gt(t,_,_);
            }
            T = itops.convolve(
                    beta, 
                    Fermion,
                    itops.vals2coefs(GKt), 
                    itops.vals2coefs(T),
                    TIME_ORDERED);
            // 7. multiply by Fbar
            for (int t = 0; t < r; t++) {
                T(t,_,_) = nda::matmul(Fbar, T(t,_,_));
            }
        }
    }
}

nda::array<dcomplex,3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad) {
    auto fc = itops.vals2coefs(f);
    auto it_eq = cppdlr::eqptsrel(n_quad+1);
    auto f_eq = nda::array<dcomplex,3>(n_quad+1, f.extent(1), f.extent(2));
    for (int i = 0; i <= n_quad; i++) {
        f_eq(i,_,_) = itops.coefs2eval(fc, it_eq(i));
    }
    return f_eq;
}

nda::array<dcomplex,3> OCA_dense(
    nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags) {

    // index orders:
    // Gt (time, N, N), where N = 2^n, n = number of orbital indices
    // Fs (num_Fs, N, N)
    // Fbars (num_Fs, r, N, N)

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
    int num_Fs = Fs.extent(0);

    // compute Fbars and Fdagbars
    auto Fdagbars = nda::array<dcomplex, 4>(num_Fs, r, N, N);
    auto Fbarsrefl = nda::array<dcomplex, 4>(num_Fs, r, N, N);
    for (int lam = 0; lam < num_Fs; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < num_Fs; nu++) {
                Fdagbars(lam,l,_,_) += hyb_coeffs(l,nu,lam)*F_dags(nu,_,_);
                Fbarsrefl(nu,l,_,_) += hyb_refl_coeffs(l,nu,lam)*Fs(lam,_,_);
            }
        }
    }

    // initialize self-energy
    nda::array<dcomplex,3> Sigma(r,N,N);
    // nda::array<dcomplex,3> Sigma_ff(r,N,N);
    // nda::array<dcomplex,3> Sigma_fb(r,N,N); // fb --> fb2 = 1, fb1 = 0, just OCA_dense_middle different
    // nda::array<dcomplex,3> Sigma_bf(r,N,N); // bf --> fb2 = 0, fb1 = 1
    // nda::array<dcomplex,3> Sigma_bb(r,N,N);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_l(r, N, N), T(r, N, N), Tmu(r, N, N), GKt(r, N, N); 
    nda::array<dcomplex, 4> Tkaps(num_Fs, r, N, N);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders
    // loop over hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
        for (int fb2 = 0; fb2 <= 1; fb2++) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F2list = (fb2==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F3list = (fb1==1) ? F_dags(_,_,_) : Fs(_,_,_);
            auto const &Fbar_array = (fb2==1) ? Fdagbars(_,_,_,_) : Fbarsrefl(_,_,_,_);
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            for (int l = 0; l < r; l++) {
                Sigma_l = 0;
                // initialize summand assoc'd with index l
                for (int lam = 0; lam < num_Fs; lam++) {
                    OCA_dense_right_in_place(
                        beta, itops, dlr_it, dlr_rf(l), (fb2==1), 
                        Gt, F2list(lam,_,_), T);
                    OCA_dense_middle_in_place(
                        (fb1==1), hyb, hyb_refl, F1list, F3list, 
                        T, Tkaps, Tmu);
                    OCA_dense_left_in_place(
                        beta, itops, dlr_it, dlr_rf(l), (fb2==1), 
                        Gt, Fbar_array(lam,l,_,_), T, GKt);
                    Sigma_l += T;
                } // sum over lambda

                // prefactor with Ks
                if (fb2 == 1) {
                    if (dlr_rf(l) <= 0) {
                        for (int t = 0; t < r; t++) {
                            Sigma_l(t,_,_) = k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t,_,_);
                        }
                        Sigma_l = Sigma_l/k_it(0, -dlr_rf(l));
                    } else {
                        Sigma_l = Sigma_l/k_it(0, dlr_rf(l));
                    }
                } else {
                    if (dlr_rf(l) >= 0) {
                        for (int t = 0; t < r; t++) {
                            Sigma_l(t,_,_) = k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t,_,_);
                        }
                        Sigma_l = Sigma_l/k_it(0, dlr_rf(l));
                    } else {
                        Sigma_l = Sigma_l/k_it(0, -dlr_rf(l));
                    }
                }
                Sigma += sfM*Sigma_l;
                // if (fb1 == 1 && fb2 == 1) Sigma_ff += sfM*Sigma_l;
                // if (fb1 == 0 && fb2 == 1) Sigma_fb += sfM*Sigma_l;
                // if (fb1 == 1 && fb2 == 0) Sigma_bf += sfM*Sigma_l;
                // if (fb1 == 0 && fb2 == 0) Sigma_bb += sfM*Sigma_l;
            } // sum over l
        } // sum over fb2
    } // sum over fb1
    return Sigma;
}

nda::array<dcomplex,3> OCA_tpz(
    nda::array_const_view<dcomplex,3> hyb,
    imtime_ops &itops, 
    double beta, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    int n_quad) {

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl); 

    // get F^dagger operators
    int num_Fs = Fs.extent(0);
    nda::array<dcomplex,3> F_dags(num_Fs, N, N);
    for (int i = 0; i < num_Fs; ++i) {
        F_dags(i,_,_) = nda::transpose(nda::conj(Fs(i,_,_)));
    }

    // get equispaced grid and evaluate functions on grid
    auto it_eq = cppdlr::eqptsrel(n_quad+1);
    nda::array<dcomplex,3> hyb_eq(n_quad+1, num_Fs, num_Fs);
    nda::array<dcomplex,3> hyb_refl_eq(n_quad+1, num_Fs, num_Fs);
    auto Gt_coeffs = itops.vals2coefs(Gt);
    nda::array<dcomplex,3> Gt_eq(n_quad+1, N, N);
    // auto hyb_eq = itops.coefs2eval(hyb, it_eq);
    for (int i = 0; i < n_quad+1; i++) {
        hyb_eq(i,_,_) = itops.coefs2eval(hyb_coeffs, it_eq(i));
        hyb_refl_eq(i,_,_) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
        // added 29 May 2025 v
        hyb_refl_eq(i,_,_) = nda::transpose(hyb_refl_eq(i,_,_));
        Gt_eq(i,_,_) = itops.coefs2eval(Gt_coeffs, it_eq(i));
    }
    nda::array<dcomplex,3> Sigma_eq(n_quad+1,N,N);

    double dt = beta/n_quad;

    for (int fb1 = 0; fb1 <= 1; fb1++) {
        for (int fb2 = 0; fb2 <= 1; fb2++) {
            // fb = 1 for forward line, else = 0
            // fb1 corresponds with line from 0 to tau_2
            auto const &F1list = (fb1==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F2list = (fb2==1) ? Fs(_,_,_) : F_dags(_,_,_);
            auto const &F3list = (fb1==1) ? F_dags(_,_,_) : Fs(_,_,_);
            auto const &F4list = (fb2==1) ? F_dags(_,_,_) : Fs(_,_,_);
            auto const &hyb1 = (fb1==1) ? hyb_eq(_,_,_) : hyb_refl_eq(_,_,_);
            auto const &hyb2 = (fb2==1) ? hyb_eq(_,_,_) : hyb_refl_eq(_,_,_);
            int sfM = (fb1^fb2) ? 1 : -1; // sign

            for (int lam = 0; lam < num_Fs; lam++) {
                for (int nu = 0; nu < num_Fs; nu++) {
                    for (int mu = 0; mu < num_Fs; mu++) {
                        for (int kap = 0; kap < num_Fs; kap++) {
                            for (int i = 1; i <= n_quad; i++) {
                                for (int i1 = 1; i1 <= i; i1++) {
                                    for (int i2 = 0; i2 <= i1; i2++) {
                                        double w = 1.0;
                                        if (i1 == i) w = w/2;
                                        if (i2 == 0 || i2 == i1) w = w/2;
                                        auto FGFGFGF = nda::matmul(F4list(nu,_,_), 
                                            nda::matmul(Gt_eq(i-i1,_,_), 
                                            nda::matmul(F3list(mu,_,_), 
                                            nda::matmul(Gt_eq(i1-i2,_,_), 
                                            nda::matmul(F2list(lam,_,_), 
                                            nda::matmul(Gt_eq(i2,_,_), F1list(kap,_,_)))))));

                                        Sigma_eq(i,_,_) += sfM*w*hyb2(i-i2,lam,nu)*hyb1(i1,mu,kap)*FGFGFGF;
                                    } // sum over i2
                                } // sum over i1 
                            } // sum over i
                        } // sum over kappa
                    } // sum over mu
                } // sum over nu
            } // sum over lambda

        } // sum over fb2
    } // sum over fb1

    Sigma_eq = dt*dt*Sigma_eq;
    
    return Sigma_eq;
}

void multiply_vertex_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::array_const_view<dcomplex,3> Fs, 
    nda::array_const_view<dcomplex,3> F_dags, 
    nda::array_const_view<dcomplex,4> Fdagbars, 
    nda::array_const_view<dcomplex,4> Fbarsrefl, 
    int v_ix, // vertex index
    int s_ix, // state index
    int l_ix, // pole index
    double pole, 
    nda::array_view<dcomplex,3> T) {

    int r = dlr_it.size();

    if (backbone.get_vertex(v_ix, 0) == 1) { // F has bar
        if (backbone.get_vertex(v_ix, 1) == 1) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fdagbars(s_ix,l_ix,_,_), T(t,_,_)); 
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fbarsrefl(s_ix,l_ix,_,_), T(t,_,_)); 
        }
    } else {
        if (backbone.get_vertex(v_ix, 1) == 1) { // F has dagger
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(F_dags(s_ix,_,_), T(t,_,_));
        } else {
            for (int t = 0; t < r; t++) T(t,_,_) = nda::matmul(Fs(s_ix,_,_), T(t,_,_));
        }
    }

    // K factor
    int bv = backbone.get_vertex(v_ix, 3); // sign on K
    if (bv != 0) {
        for (int t = 0; t < r; t++) {
            T(t,_,_) = k_it(dlr_it(t), bv * pole) * T(t,_,_);
        }
    }
}

void compute_edge_dense(
    BackboneSignature& backbone, 
    nda::vector_const_view<double> dlr_it, 
    nda::vector_const_view<double> dlr_rf, 
    nda::array_const_view<dcomplex,3> Gt, 
    int e_ix, // edge index
    nda::array_view<dcomplex,3> GKt) {

    GKt = Gt; 
    int m = backbone.m; 
    int r = dlr_it.size(); 
    for (int x = 0; x < m-1; x++) {
        int be = backbone.get_edge(e_ix, x); // sign on K
        if (be != 0) {
            for (int t = 0; t < r; t++) {
                GKt(t,_,_) = k_it(dlr_it(t), be * dlr_rf(backbone.get_pole_ind(x))) * GKt(t,_,_);
            }
        }
    }
}

void hyb_vertex(
    BackboneSignature& backbone, 
    nda::array_const_view<dcomplex,3> hyb, 
    nda::array_const_view<dcomplex,3> F0s, 
    nda::array_const_view<dcomplex,3> Fhybs, 
    nda::array_view<dcomplex,4> Tkaps, 
    nda::array_view<dcomplex,3> Tmu, 
    nda::array_view<dcomplex,3> T) {

    int n = backbone.n; 
    int r = hyb.extent(0); 
    for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) {
            Tkaps(kap,t,_,_) = nda::matmul(T(t,_,_), F0s(kap,_,_));
        }
    }
    for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
            for (int t = 0; t < r; t++) {
                Tmu(t,_,_) += hyb(t,mu,kap)*Tkaps(kap,t,_,_);
            }
        }
        for (int t = 0; t < r; t++) {
            T(t,_,_) += nda::matmul(Fhybs(mu,_,_), Tmu(t,_,_)); // ??? check if just = ? 
        }
    }
}

nda::array<dcomplex, 3> eval_backbone_dense(BackboneSignature &backbone, 
    double beta, 
    imtime_ops &itops, 
    nda::array_const_view<dcomplex, 3> hyb, 
    nda::array_const_view<dcomplex, 3> Gt, 
    nda::array_const_view<dcomplex, 3> Fs, 
    nda::array_const_view<dcomplex, 3> F_dags) {

    // index orders:
    // Gt (time, N, N), where N = 2^n, n = number of orbital indices
    // Fs (n, N, N)
    // Fbars (n, r, N, N)

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);
    int m = backbone.m;

    auto hyb_coeffs = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
    int n = Fs.extent(0);

    // compute Fbars and Fdagbars
    auto Fdagbars = nda::array<dcomplex, 4>(n, r, N, N);
    auto Fbarsrefl = nda::array<dcomplex, 4>(n, r, N, N);
    for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < r; l++) {
            for (int nu = 0; nu < n; nu++) {
                Fdagbars(lam,l,_,_) += hyb_coeffs(l,nu,lam)*F_dags(nu,_,_);
                Fbarsrefl(nu,l,_,_) += hyb_refl_coeffs(l,nu,lam)*Fs(lam,_,_);
            }
        }
    }

    // initialize self-energy
    nda::array<dcomplex,3> Sigma(r,N,N);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_L(r, N, N), T(r, N, N), Tmu(r, N, N), GKt(r, N, N); 
    nda::array<dcomplex, 4> Tkaps(n, r, N, N);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders
    // loop over hybridization lines

    nda::vector<int> fb_vec(m), states(2*m);
    auto pole_inds = nda::zeros<int>(m-1);

    for (int fb = 0; fb < pow(2,m); fb++) { // loop over 2^m combos of for/backward lines
        int fb0 = fb;
        // turn (int) fb into a vector of 1s and 0s corresp. to forward, backward lines, resp. 
        for (int i = 0; i < m; i++) {fb_vec(i) = fb0 % 2; fb0 = fb0 / 2;}
        backbone.set_directions(fb_vec); // give line directions to backbone object
        int sign = (fb==2 || fb==0) ? 1 : -1;  // (fb_vec(0)^fb_vec(1)) ? -1 : 1; // TODO: figure this out
        
        // L = pole multiindex
        for (int L = 0; L < pow(r,m-1); L++) { // loop over all combinations of pole indices
            Sigma_L = 0; 

            int L0 = L;
            // turn (int) L into a vector of pole indices
            for (int i = 0; i < m-1; i++) {pole_inds(i) = L0 % r; L0 = L0 / r;}
            backbone.set_pole_inds(pole_inds, dlr_rf); // give pole indices to backbone object

            // print a negative and positive pole diagram for each set of line directions
            if (L == 0) std::cout << "fb = " << fb << " " << backbone << std::endl;
            if (L == r-1) std::cout << "fb = " << fb << " " << backbone << std::endl;

            // set up state (Greek) indices that are explicity summed over
            for (int s = 0; s < pow(n, m-1); s++) { // loop over all combos of states
                int s0 = s;
                // turn (int) s into a vector of state indices
                for (int i = 1; i < m; i++) { // loop over lines, skipping the one connected to zero
                    states(backbone.get_topology(i, 0)) = s0 % n;
                    states(backbone.get_topology(i, 1)) = s0 % n;
                    // state indices on vertices connected by a line are the same
                    s0 = s0 / n; 
                }

                // 1. Starting from tau_1, proceed right to left, performing 
                //    multiplications at vertices and convolutions at edges, 
                //    until reaching the vertex containing the undecomposed 
                //    hybridization line Delta_{mu kappa}
                T = Gt; // T stores the result as a move right to left
                // T is initialized with Gt, which is always the function at the rightmost edge
                for (int v = 1; v < backbone.get_topology(0,1); v++) { // loop from the first vertex to before the special vertex
                    // compute vertex (multiply)
                    int l = pole_inds(backbone.get_vertex(v, 2)); // get value of pole index assoc'd with this vertex
                    double pole = dlr_rf(l); 
                    multiply_vertex_dense(backbone, dlr_it, Fs, F_dags, Fdagbars, Fbarsrefl, v, states(v), l, pole, T); 
                    compute_edge_dense(backbone, dlr_it, dlr_rf, Gt, v, GKt); 
                    // convolve with edge
                    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED); 
                } 

                // 2. For each kappa, multiply by F_kappa(^dag). Then for each 
                //    mu, kappa, multiply by Delta_{mu kappa}, and sum over 
                //    kappa. Finally for each mu, multiply F_mu[^dag] and sum 
                //    over mu. 
                if (backbone.get_vertex(0, 1) == 0) { // line connected to zero is forward
                    hyb_vertex(backbone, hyb, Fs, F_dags, Tkaps, Tmu, T); 
                } else { // line connected to zero is backward
                    hyb_vertex(backbone, hyb_refl, F_dags, Fs, Tkaps, Tmu, T); 
                }

                // 3. Continue right to left until the final vertex 
                //    multiplication is complete.
                for (int v = backbone.get_topology(0,1) + 1; v < 2*m; v++) { // loop from the next edge to the final vertex
                    compute_edge_dense(backbone, dlr_it, dlr_rf, Gt, v-1, GKt);
                    T = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED); 
                    int l = pole_inds(backbone.get_vertex(v, 2)); // get the value of pole index assoc'd with this vertex
                    double pole = dlr_rf(l); 
                    multiply_vertex_dense(backbone, dlr_it, Fs, F_dags, Fdagbars, Fbarsrefl, v, states(v), l, pole, T); 
                }
                Sigma_L += T; 
            } // sum over states

            // 4. Multiply by prefactor
            for (int p = 0; p < m-1; p++) { // loop over pole indices
                int exp = backbone.get_prefactor(p, 2); // exponent on K for this pole index
                if (exp != 0) { 
                    int Ksign = backbone.get_prefactor(p, 1); // 1 if K^+, -1 if K^-
                    double om = dlr_rf(pole_inds(p)); 
                    double k = k_it(0, Ksign*om); 
                    for (int q = 0; q < exp; q++) Sigma_L = Sigma_L / k; 
                }
                Sigma_L = Sigma_L * backbone.get_prefactor(p, 0); // multiply by overall sign on prefactor
            }
            Sigma += sign*Sigma_L; 
            backbone.reset_pole_inds(); 
        } // sum over poles
        backbone.reset_directions(); 
    } // sum over forward/backward lines

    return Sigma;
}
