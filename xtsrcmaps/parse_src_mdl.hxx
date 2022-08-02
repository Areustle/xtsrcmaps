#pragma once
#include <string>
#include <vector>

#include <xtsrcmaps/source.hxx>

namespace Fermi
{

auto
parse_src_xml(std::string const& src_file_name) -> std::vector<Source>;

} // namespace Fermi
