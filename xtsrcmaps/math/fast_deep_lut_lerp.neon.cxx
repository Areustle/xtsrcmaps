#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>

void
inline
multiply_and_add_neon(const float               scalar1,
                      const float               scalar2,
                      const std::vector<float>& array1,
                      const std::vector<float>& array2,
                      std::vector<float>&       result) {
    size_t const N           = array1.size();
    float32x4_t  scalar_vec1 = vdupq_n_f32(scalar1);
    float32x4_t  scalar_vec2 = vdupq_n_f32(scalar2);

    size_t i                 = 0;
    for (; i + 4 <= N; i += 4) {
        float32x4_t vec1 = vld1q_f32(&array1[i]);
        float32x4_t vec2 = vld1q_f32(&array2[i]);
        float32x4_t res1 = vmulq_f32(vec1, scalar_vec1);
        float32x4_t res2 = vmulq_f32(vec2, scalar_vec2);
        float32x4_t sum  = vaddq_f32(res1, res2);
        vst1q_f32(&result[i], sum);
    }

    for (; i < N; ++i) {
        result[i] = scalar1 * array1[i] + scalar2 * array2[i];
    }
}

void inline foo(float32x4_t v, int const n){
    // Extract the nth value from the vector
    float nth_value;
    switch (n) {
        case 0: nth_value = vgetq_lane_f32(v, 0); break;
        case 1: nth_value = vgetq_lane_f32(v, 1); break;
        case 2: nth_value = vgetq_lane_f32(v, 2); break;
        case 3: nth_value = vgetq_lane_f32(v, 3); break;
        default: nth_value = 0.0f; // This should never happen
    }
    // Create a vector with all elements set to the nth value
    float32x4_t nth_value_vec = vdupq_n_f32(nth_value);
    // Compare the original vector with the nth value vector to create the mask
    uint32x4_t mask = vceqq_f32(v, nth_value_vec);
}


#endif
