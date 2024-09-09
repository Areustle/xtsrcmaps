#include "xtsrcmaps/config/config.hxx"

namespace Fermi::Config {
auto parse_cli_to_cfg(int const argc, char** argv) -> Config::XtCfg;
} // namespace Fermi::Config
