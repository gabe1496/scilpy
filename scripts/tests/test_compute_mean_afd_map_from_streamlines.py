#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_mean_afd_map_from_streamlines', '--help')
    assert ret.success
