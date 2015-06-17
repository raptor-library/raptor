#!/usr/bin/env python

VERSION = '0.0.1'
APPNAME = 'raptor'

top = '.'
opt = 'build'

def options(opt):
    opt.load('compiler_cxx waf_unit_test')


def configure(cfg):
    cfg.load('compiler_cxx waf_unit_test')
    cfg.env.append_value('CXXFLAGS', ['-O2', '--std=c++11'])


def build(bld):
    bld.recurse('raptor')
    bld.recurse('external')
    from waflib.Tools import waf_unit_test
    bld.add_post_fun(waf_unit_test.summary)
