program test
  use, intrinsic :: iso_c_binding, only: c_int
  implicit none

  integer(c_int) :: vec_handle
  interface
     function create_vec(n) bind(C, name='create_vec') result(handle)
       use, intrinsic :: iso_c_binding, only: c_int
       integer(c_int), value :: n
       integer(c_int) :: handle
     end function create_vec


     function getsize(handle) bind(C, name='getsize') result(size)
       use, intrinsic :: iso_c_binding, only: c_int
       integer(c_int), value :: handle
       integer(c_int) :: size
     end function getsize
  end interface

  vec_handle = create_vec(11)
  print*, 'size', getsize(vec_handle)
end program test
