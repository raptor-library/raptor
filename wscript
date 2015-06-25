#!/usr/bin/env python

APPNAME = 'raptor'
VERSION = '0.0.1'

top = '.'
opt = 'build'


def options(opt):
    opt.load('compiler_cxx')
    opt.add_option('--shared', action='store_true',
                   help='build a shared library')
    opt.add_option('--static', action='store_true',
                   help='build a static library')


def configure(cfg):
    cfg.load('compiler_cxx')
    cfg.env.append_value('CXXFLAGS', ['-O2', '--std=c++11'])


def build(bld):
    bld.recurse('raptor')
    bld.recurse('external')
    # bld.recurse('examples')
