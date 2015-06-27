function(add_src_promisc dir)
  FILE(GLOB_RECURSE ${dir}_sources_ ${dir}/*.cpp)
  SET(${dir}_sources ${${dir}_sources_} PARENT_SCOPE)
endfunction(add_src_promisc)
