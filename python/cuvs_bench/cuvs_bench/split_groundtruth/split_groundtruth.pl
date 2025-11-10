#!/usr/bin/perl

# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

use warnings;
use strict;
use autodie qw(open close);


@ARGV == 2
  or die "usage: $0 input output_prefix\n";

open my $fh, '<:raw', $ARGV[0];

my $raw;
read($fh, $raw, 8);
my ($nrows, $dim) = unpack('LL', $raw);

my $expected_size = 8 + $nrows * $dim * (4 + 4);
my $size = (stat($fh))[7];
$size == $expected_size
  or die("error: expected size is $expected_size, but actual size is $size\n");


open my $fh_out1, '>:raw', "$ARGV[1].neighbors.ibin";
open my $fh_out2, '>:raw', "$ARGV[1].distances.fbin";

print {$fh_out1} $raw;
print {$fh_out2} $raw;

read($fh, $raw, $nrows * $dim * 4);
print {$fh_out1} $raw;
read($fh, $raw, $nrows * $dim * 4);
print {$fh_out2} $raw;
