#include <iostream>
#include "mhsa.h"
#include <fstream>  // Enables file writing


int main() {

    // INPUT  
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\formatted_input_tensor_196x64.txt"
    

    // Q WEIGHTS
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wq_head1.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wq_head2.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wq_head3.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wq_head4.txt"

    // K WEIGHTS
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wk_head1.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wk_head2.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wk_head3.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wk_head4.txt"

    // V WEIGHTS
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wv_head1.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wv_head2.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wv_head3.txt"
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wv_head4.txt"
    
    // OUTPUT PROJECTION WEIGHT
    #include "C:\Users\prane_t0yv1to\Downloads\NIT_W_SUMMER_PROJECT\formatted_kaggle_data\wo.txt"
    

    // Input and output
    data_t output[NUM_PATCHES][EMBEDDING_DIM];


    // Call MHSA 
    multi_head_attention(
        input_tensor_196x64,

        wq_head1, wq_head2, wq_head3, wq_head4,
        wk_head1, wk_head2, wk_head3, wk_head4,
        wv_head1, wv_head2, wv_head3, wv_head4,

        wo, 
        
        output
    );


    // Write output to file
    std::ofstream outfile("C:/Users/prane_t0yv1to/Downloads/NIT_W_SUMMER_PROJECT/HLS_mhsa_output_196x64.txt");

    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file!\n";
        return 1;
    }

    for (int i = 0; i < NUM_PATCHES; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            outfile << output[i][j];
            if (j != EMBEDDING_DIM - 1) outfile << ", ";
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Output saved to C:/Users/prane_t0yv1to/Downloads/NIT_W_SUMMER_PROJECT/HLS_mhsa_output_196x64.txt \n";


    return 0;
}
