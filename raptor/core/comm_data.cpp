// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "core/comm_data.hpp"

namespace raptor 
{

template<>
std::vector<double>& CommData::get_buffer<double>()
{
    return buffer;
}

template<>
std::vector<int>& CommData::get_buffer<int>()
{
    return int_buffer;
}

}
