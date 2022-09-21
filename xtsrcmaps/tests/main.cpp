#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "xtsrcmaps/fitsfuncs.hxx"

int factorial(int number) { return number <= 1 ? number : factorial(number - 1) * number; }

TEST_CASE("testing the factorial function") {
    CHECK(factorial(1) == 1);
    CHECK(factorial(2) == 2);
    CHECK(factorial(3) == 6);
    CHECK(factorial(10) == 3628800);
}

TEST_CASE("fermi read irf_pars") {
  auto ov = Fermi::fits::read_irf_pars("/Users/areustle/nasa/fermi/xtsrcmaps/analysis/psf_P8R3_SOURCE_V2_FB.fits", "RPSF_FRONT");
  CHECK(ov);

  auto v = ov.value();

  CHECK(v.extents.size() == 10);
  CHECK(v.extents == std::vector<size_t>{23, 23, 8, 8, 184, 184, 184, 184, 184, 184});

  CHECK(v.rowdata.size() == 1);
  CHECK(v.rowdata[0].size() == 1166);

  // auto expect = std::vector<float>{};
  // CHECK(v.rowdata[0] == );
}
