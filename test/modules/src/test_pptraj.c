// File under test pptraj.h and pptraj_compressed.h
#include "pptraj.h"
#include "pptraj_compressed.h"

#include <stdlib.h>
#include <string.h>

#include "unity.h"

// #define SHOW_OUTPUT

struct poly4d figure8_pieces[] = {
  {
    .p = {
      { 0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493 },
      { -0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 1.05
  },
  {
    .p = {
      { 0.396058, 0.918033, 0.128965, -0.773546, 0.339704, 0.034310, -0.026417, -0.030049 },
      { -0.445604, -0.684403, 0.888433, 1.493630, -1.361618, -0.139316, 0.158875, 0.095799 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.71
  },
  {
    .p = {
      { 0.922409, 0.405715, -0.582968, -0.092188, -0.114670, 0.101046, 0.075834, -0.037926 },
      { -0.291165, 0.967514, 0.421451, -1.086348, 0.545211, 0.030109, -0.050046, -0.068177 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.62
  },
  {
    .p = {
      { 0.923174, -0.431533, -0.682975, 0.177173, 0.319468, -0.043852, -0.111269, 0.023166 },
      { 0.289869, 0.724722, -0.512011, -0.209623, -0.218710, 0.108797, 0.128756, -0.055461 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.7
  },
  {
    .p = {
      { 0.405364, -0.834716, 0.158939, 0.288175, -0.373738, -0.054995, 0.036090, 0.078627 },
      { 0.450742, -0.385534, -0.954089, 0.128288, 0.442620, 0.055630, -0.060142, -0.076163 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.56
  },
  {
    .p = {
      { 0.001062, -0.646270, -0.012560, -0.324065, 0.125327, 0.119738, 0.034567, -0.063130 },
      { 0.001593, -1.031457, 0.015159, 0.820816, -0.152665, -0.130729, -0.045679, 0.080444 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.56
  },
  {
    .p = {
      { -0.402804, -0.820508, -0.132914, 0.236278, 0.235164, -0.053551, -0.088687, 0.031253 },
      { -0.449354, -0.411507, 0.902946, 0.185335, -0.239125, -0.041696, 0.016857, 0.016709 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.7
  },
  {
    .p = {
      { -0.921641, -0.464596, 0.661875, 0.286582, -0.228921, -0.051987, 0.004669, 0.038463 },
      { -0.292459, 0.777682, 0.565788, -0.432472, -0.060568, -0.082048, -0.009439, 0.041158 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.62
  },
  {
    .p = {
      { -0.923935, 0.447832, 0.627381, -0.259808, -0.042325, -0.032258, 0.001420, 0.005294 },
      { 0.288570, 0.873350, -0.515586, -0.730207, -0.026023, 0.288755, 0.215678, -0.148061 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 0.71
  },
  {
    .p = {
      { -0.398611, 0.850510, -0.144007, -0.485368, -0.079781, 0.176330, 0.234482, -0.153567 },
      { 0.447039, -0.532729, -0.855023, 0.878509, 0.775168, -0.391051, -0.713519, 0.391628 },
      { 0, 0, 0, 0, 0, 0, 0, 0 },
      { 0, 0, 0, 0, 0, 0, 0, 0 }
    },
    .duration = 1.05
  },
};

const uint8_t figure8_compressed_pieces[] = {
  /* initial position */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

  /* t = 0.000s */
  0x0f, 0x1a, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x7f, 0x00, 0x02, 0x01, 0x8c,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd1, 0xff, 0x3e, 0xff, 0xa9, 0xfe, 0x42, 0xfe,

  /* t = 1.050s */
  0x0f, 0xc6, 0x02, 0xe9, 0x01, 0x49, 0x02, 0xa5, 0x02, 0xf6, 0x02, 0x3a, 0x03, 0x71, 0x03, 0x9a,
  0x03, 0xfd, 0xfd, 0xcd, 0xfd, 0xc1, 0xfd, 0xe0, 0xfd, 0x23, 0xfe, 0x7b, 0xfe, 0xdd, 0xfe,

  /* t = 1.760s */
  0x0f, 0x6c, 0x02, 0xbe, 0x03, 0xd8, 0x03, 0xe6, 0x03, 0xe7, 0x03, 0xdb, 0x03, 0xc1, 0x03, 0x9b,
  0x03, 0x33, 0xff, 0x90, 0xff, 0xee, 0xff, 0x47, 0x00, 0x98, 0x00, 0xe2, 0x00, 0x22, 0x01,

  /* t = 2.380s */
  0x0f, 0xbc, 0x02, 0x70, 0x03, 0x35, 0x03, 0xec, 0x02, 0x98, 0x02, 0x40, 0x02, 0xe9, 0x01, 0x95,
  0x01, 0x6a, 0x01, 0xa7, 0x01, 0xd5, 0x01, 0xf2, 0x01, 0xfa, 0x01, 0xe9, 0x01, 0xc3, 0x01,

  /* t = 3.080s */
  0x0f, 0x30, 0x02, 0x53, 0x01, 0x12, 0x01, 0xd6, 0x00, 0x9d, 0x00, 0x68, 0x00, 0x35, 0x00, 0x01,
  0x00, 0xa4, 0x01, 0x77, 0x01, 0x3c, 0x01, 0xf6, 0x00, 0xa7, 0x00, 0x54, 0x00, 0x02, 0x00,

  /* t = 3.640s */
  0x0f, 0x30, 0x02, 0xcd, 0xff, 0x99, 0xff, 0x64, 0xff, 0x2b, 0xff, 0xee, 0xfe, 0xaf, 0xfe, 0x6d,
  0xfe, 0xaf, 0xff, 0x5d, 0xff, 0x0f, 0xff, 0xc9, 0xfe, 0x8e, 0xfe, 0x60, 0xfe, 0x3f, 0xfe,

  /* t = 4.200s */
  0x0f, 0xbc, 0x02, 0x1b, 0xfe, 0xc6, 0xfd, 0x70, 0xfd, 0x1d, 0xfd, 0xd3, 0xfc, 0x95, 0xfc, 0x66,
  0xfc, 0x15, 0xfe, 0x01, 0xfe, 0x04, 0xfe, 0x1e, 0xfe, 0x4d, 0xfe, 0x8e, 0xfe, 0xdc, 0xfe,

  /* t = 4.900s */
  0x0f, 0x6c, 0x02, 0x3d, 0xfc, 0x20, 0xfc, 0x11, 0xfc, 0x11, 0xfc, 0x20, 0xfc, 0x3c, 0xfc, 0x64,
  0xfc, 0x20, 0xff, 0x70, 0xff, 0xc6, 0xff, 0x21, 0x00, 0x7c, 0x00, 0xd3, 0x00, 0x21, 0x01,

  /* t = 5.520s */
  0x0f, 0xc6, 0x02, 0x91, 0xfc, 0xce, 0xfc, 0x17, 0xfd, 0x69, 0xfd, 0xc1, 0xfd, 0x1b, 0xfe, 0x71,
  0xfe, 0x79, 0x01, 0xc5, 0x01, 0xfe, 0x01, 0x1b, 0x02, 0x17, 0x02, 0xf5, 0x01, 0xbf, 0x01,

  /* t = 6.230s */
  0x0f, 0x1d, 0x04, 0xf1, 0xfe, 0x6a, 0xff, 0xca, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x6f, 0x01, 0xf2, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

  /* t =  7.28s */
  0x00, 0x00, 0x00
};

const uint8_t simplified_figure8_compressed_pieces[] = {
  /* initial position */
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

  /* trajectory */
  0x10, 0xd0, 0x07, 0xdc, 0x05, 0x05, 0xd0, 0x07, 0xe8, 0x03, 0xe8, 0x03, 0x05, 0xd0, 0x07, 0xd0,
  0x07, 0x00, 0x00, 0x05, 0xd0, 0x07, 0xe8, 0x03, 0x18, 0xfc, 0x05, 0xd0, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x05, 0xd0, 0x07, 0x18, 0xfc, 0xe8, 0x03, 0x05, 0xd0, 0x07, 0x30, 0xf8, 0x00, 0x00, 0x05,
  0xd0, 0x07, 0x18, 0xfc, 0x18, 0xfc, 0x05, 0xd0, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10, 0xd0, 0x07,
  0x00, 0x00, 0x00, 0x00, 0x00,
};

const uint8_t frame_compressed_pieces[] = {
  /* initial position */
  0xb8, 0x0b, 0x58, 0x1b, 0xe8, 0x03, 0x00, 0x00,

  /* trajectory */
  0x00, 0xf4, 0x01, 0x00, 0xf4, 0x01, 0x04, 0xf4, 0x01, 0xa8, 0x1b, 0x04, 0xf4, 0x01, 0xb4, 0x1e,
  0x04, 0xf4, 0x01, 0xc4, 0x22, 0x04, 0xf4, 0x01, 0xfc, 0x26, 0x04, 0xf4, 0x01, 0xda, 0x2a, 0x14,
  0xf4, 0x01, 0xec, 0x2c, 0xf2, 0x03, 0x10, 0xf4, 0x01, 0x96, 0x05, 0x10, 0xf4, 0x01, 0xf8, 0x07,
  0x10, 0xf4, 0x01, 0x8c, 0x0a, 0x10, 0xf4, 0x01, 0x20, 0x0d, 0x10, 0xf4, 0x01, 0x78, 0x0f, 0x14,
  0xf4, 0x01, 0xe2, 0x2c, 0xcc, 0x10, 0x04, 0xf4, 0x01, 0x30, 0x2a, 0x04, 0xf4, 0x01, 0xa2, 0x26,
  0x04, 0xf4, 0x01, 0xd8, 0x22, 0x04, 0xf4, 0x01, 0xfa, 0x1e, 0x04, 0xf4, 0x01, 0x08, 0x1b, 0x04,
  0xf4, 0x01, 0x16, 0x17, 0x04, 0xf4, 0x01, 0x38, 0x13, 0x04, 0xf4, 0x01, 0x78, 0x0f, 0x04, 0xf4,
  0x01, 0xfe, 0x0b, 0x14, 0xf4, 0x01, 0xc4, 0x09, 0xc2, 0x10, 0x10, 0xf4, 0x01, 0x1e, 0x0f, 0x10,
  0xf4, 0x01, 0xbc, 0x0c, 0x10, 0xf4, 0x01, 0x28, 0x0a, 0x10, 0xf4, 0x01, 0x94, 0x07, 0x10, 0xf4,
  0x01, 0x3c, 0x05, 0x14, 0xf4, 0x01, 0xce, 0x09, 0xe8, 0x03, 0x04, 0xf4, 0x01, 0xea, 0x0b, 0x04,
  0xf4, 0x01, 0xf6, 0x0e, 0x04, 0xf4, 0x01, 0x48, 0x12, 0x04, 0xf4, 0x01, 0x9a, 0x15, 0x04, 0xf4,
  0x01, 0xba, 0x18, 0x04, 0xf4, 0x01, 0x26, 0x1b, 0x14, 0xf4, 0x01, 0x58, 0x1b, 0x02, 0x03, 0x10,
  0xf4, 0x01, 0x3c, 0x00, 0x10, 0xf4, 0x01, 0xf0, 0x00, 0x14, 0xf4, 0x01, 0x9e, 0x1b, 0x8a, 0x02,
  0x14, 0xf4, 0x01, 0x8e, 0x1c, 0xde, 0x03, 0x14, 0xf4, 0x01, 0xd2, 0x1e, 0x4c, 0x04, 0x14, 0xf4,
  0x01, 0xd4, 0x21, 0x1a, 0x04, 0x14, 0xf4, 0x01, 0xd6, 0x24, 0xe8, 0x03, 0x14, 0xf4, 0x01, 0x9c,
  0x27, 0xf2, 0x03, 0x14, 0xf4, 0x01, 0xfe, 0x29, 0x60, 0x04, 0x14, 0xf4, 0x01, 0xa2, 0x2b, 0x82,
  0x05, 0x14, 0xf4, 0x01, 0x88, 0x2c, 0x4e, 0x07, 0x14, 0xf4, 0x01, 0xba, 0x2c, 0x74, 0x09, 0x14,
  0xf4, 0x01, 0x38, 0x2c, 0x90, 0x0b, 0x14, 0xf4, 0x01, 0x2a, 0x2b, 0x70, 0x0d, 0x14, 0xf4, 0x01,
  0xb8, 0x29, 0xe2, 0x0e, 0x14, 0xf4, 0x01, 0xce, 0x27, 0xe6, 0x0f, 0x14, 0xf4, 0x01, 0x3a, 0x25,
  0x90, 0x10, 0x14, 0xf4, 0x01, 0x4c, 0x22, 0xf4, 0x10, 0x14, 0xf4, 0x01, 0x36, 0x1f, 0x30, 0x11,
  0x14, 0xf4, 0x01, 0x02, 0x1c, 0x4e, 0x11, 0x14, 0xf4, 0x01, 0xce, 0x18, 0x44, 0x11, 0x14, 0xf4,
  0x01, 0x9a, 0x15, 0x1c, 0x11, 0x14, 0xf4, 0x01, 0x8e, 0x12, 0xcc, 0x10, 0x14, 0xf4, 0x01, 0xaa,
  0x0f, 0x54, 0x10, 0x14, 0xf4, 0x01, 0x3e, 0x0d, 0xa0, 0x0f, 0x14, 0xf4, 0x01, 0x7c, 0x0b, 0x7e,
  0x0e, 0x14, 0xf4, 0x01, 0x50, 0x0a, 0xf8, 0x0c, 0x14, 0xf4, 0x01, 0xce, 0x09, 0x18, 0x0b, 0x14,
  0xf4, 0x01, 0xf6, 0x09, 0xfc, 0x08, 0x14, 0xf4, 0x01, 0xc8, 0x0a, 0xf4, 0x06, 0x14, 0xf4, 0x01,
  0x30, 0x0c, 0x46, 0x05, 0x14, 0xf4, 0x01, 0xe8, 0x0d, 0x56, 0x04, 0x14, 0xf4, 0x01, 0xe6, 0x0f,
  0xfc, 0x03, 0x14, 0xf4, 0x01, 0xf8, 0x11, 0xe8, 0x03, 0x14, 0xf4, 0x01, 0x1e, 0x14, 0x06, 0x04,
  0x14, 0xf4, 0x01, 0x44, 0x16, 0x2e, 0x04, 0x14, 0xf4, 0x01, 0x6a, 0x18, 0x42, 0x04, 0x14, 0xf4,
  0x01, 0x40, 0x1a, 0xb6, 0x03, 0x14, 0xf4, 0x01, 0x08, 0x1b, 0xf4, 0x01, 0x14, 0xf4, 0x01, 0x58,
  0x1b, 0x00, 0x00, 0x15, 0xf4, 0x01, 0x90, 0x0b, 0x6c, 0x1b, 0xa0, 0x00, 0x15, 0xf4, 0x01, 0x36,
  0x0b, 0xa8, 0x1b, 0xfe, 0x01, 0x15, 0xf4, 0x01, 0x78, 0x0a, 0x20, 0x1c, 0xca, 0x03, 0x15, 0xf4,
  0x01, 0x10, 0x09, 0xfc, 0x1c, 0x8c, 0x05, 0x15, 0xf4, 0x01, 0x62, 0x07, 0xce, 0x1d, 0xbe, 0x05,
  0x15, 0xf4, 0x01, 0xd2, 0x05, 0xaa, 0x1e, 0x64, 0x05, 0x15, 0xf4, 0x01, 0x60, 0x04, 0x90, 0x1f,
  0xf6, 0x04, 0x15, 0xf4, 0x01, 0x02, 0x03, 0xa8, 0x20, 0x88, 0x04, 0x15, 0xf4, 0x01, 0xcc, 0x01,
  0xf2, 0x21, 0x24, 0x04, 0x15, 0xf4, 0x01, 0xc8, 0x00, 0xa0, 0x23, 0xe8, 0x03, 0x15, 0xf4, 0x01,
  0x14, 0x00, 0xe4, 0x25, 0x06, 0x04, 0x15, 0xf4, 0x01, 0x6e, 0x00, 0x0e, 0x29, 0xce, 0x04, 0x15,
  0xf4, 0x01, 0x76, 0x02, 0x16, 0x2b, 0xaa, 0x05, 0x15, 0xf4, 0x01, 0x32, 0x05, 0x42, 0x2c, 0x86,
  0x06, 0x15, 0xf4, 0x01, 0x34, 0x08, 0xc4, 0x2c, 0x76, 0x07, 0x15, 0xf4, 0x01, 0x36, 0x0b, 0xba,
  0x2c, 0x70, 0x08, 0x15, 0xf4, 0x01, 0x10, 0x0e, 0x38, 0x2c, 0x88, 0x09, 0x15, 0xf4, 0x01, 0x9a,
  0x10, 0x16, 0x2b, 0xbe, 0x0a, 0x15, 0xf4, 0x01, 0x98, 0x12, 0x40, 0x29, 0x30, 0x0c, 0x15, 0xf4,
  0x01, 0x92, 0x13, 0x66, 0x26, 0xfc, 0x0d, 0x15, 0xf4, 0x01, 0x1a, 0x13, 0x6e, 0x23, 0x82, 0x0f,
  0x15, 0xf4, 0x01, 0x02, 0x12, 0x66, 0x21, 0x54, 0x10, 0x15, 0xf4, 0x01, 0xae, 0x10, 0xc2, 0x1f,
  0xd6, 0x10, 0x15, 0xf4, 0x01, 0x32, 0x0f, 0x50, 0x1e, 0x26, 0x11, 0x15, 0xf4, 0x01, 0xac, 0x0d,
  0xfc, 0x1c, 0x4e, 0x11, 0x15, 0xf4, 0x01, 0x12, 0x0c, 0xbc, 0x1b, 0x62, 0x11, 0x15, 0xf4, 0x01,
  0x78, 0x0a, 0x90, 0x1a, 0x58, 0x11, 0x15, 0xf4, 0x01, 0xde, 0x08, 0x64, 0x19, 0x44, 0x11, 0x15,
  0xf4, 0x01, 0x3a, 0x07, 0x42, 0x18, 0x12, 0x11, 0x15, 0xf4, 0x01, 0xaa, 0x05, 0x0c, 0x17, 0xcc,
  0x10, 0x15, 0xf4, 0x01, 0x24, 0x04, 0xd6, 0x15, 0x72, 0x10, 0x15, 0xf4, 0x01, 0xb2, 0x02, 0x82,
  0x14, 0x04, 0x10, 0x15, 0xf4, 0x01, 0x68, 0x01, 0xfc, 0x12, 0x6e, 0x0f, 0x15, 0xf4, 0x01, 0x64,
  0x00, 0x30, 0x11, 0xb0, 0x0e, 0x15, 0xf4, 0x01, 0x32, 0x00, 0x92, 0x0e, 0x84, 0x0d, 0x15, 0xf4,
  0x01, 0x68, 0x01, 0x6c, 0x0c, 0x6c, 0x0c, 0x15, 0xf4, 0x01, 0x84, 0x03, 0x0e, 0x0b, 0x86, 0x0b,
  0x15, 0xf4, 0x01, 0x18, 0x06, 0x32, 0x0a, 0xbe, 0x0a, 0x15, 0xf4, 0x01, 0xf2, 0x08, 0xd8, 0x09,
  0xf6, 0x09, 0x15, 0xf4, 0x01, 0xea, 0x0b, 0xf6, 0x09, 0x2e, 0x09, 0x15, 0xf4, 0x01, 0xce, 0x0e,
  0x8c, 0x0a, 0x52, 0x08, 0x15, 0xf4, 0x01, 0x62, 0x11, 0xc2, 0x0b, 0x58, 0x07, 0x15, 0xf4, 0x01,
  0x38, 0x13, 0xe8, 0x0d, 0xfa, 0x05, 0x15, 0xf4, 0x01, 0x6a, 0x13, 0x12, 0x11, 0x74, 0x04, 0x15,
  0xf4, 0x01, 0xf2, 0x12, 0x1a, 0x13, 0xf2, 0x03, 0x15, 0xf4, 0x01, 0x5c, 0x12, 0x96, 0x14, 0xde,
  0x03, 0x15, 0xf4, 0x01, 0xa8, 0x11, 0xcc, 0x15, 0xfc, 0x03, 0x15, 0xf4, 0x01, 0xea, 0x10, 0xd0,
  0x16, 0x42, 0x04, 0x15, 0xf4, 0x01, 0x22, 0x10, 0xac, 0x17, 0x92, 0x04, 0x15, 0xf4, 0x01, 0x50,
  0x0f, 0x74, 0x18, 0xe2, 0x04, 0x15, 0xf4, 0x01, 0x74, 0x0e, 0x32, 0x19, 0x28, 0x05, 0x05, 0xf4,
  0x01, 0x84, 0x0d, 0xdc, 0x19, 0x15, 0xf4, 0x01, 0x9e, 0x0c, 0x90, 0x1a, 0xde, 0x03, 0x15, 0xf4,
  0x01, 0x30, 0x0c, 0xea, 0x1a, 0x94, 0x02, 0x15, 0xf4, 0x01, 0xf4, 0x0b, 0x1c, 0x1b, 0x90, 0x01,
  0x15, 0xf4, 0x01, 0xd6, 0x0b, 0x3a, 0x1b, 0xd2, 0x00, 0x15, 0xf4, 0x01, 0xc2, 0x0b, 0x4e, 0x1b,
  0x46, 0x00, 0x15, 0xf4, 0x01, 0xb8, 0x0b, 0x58, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x00,
};

void setUp(void)
{
  // Empty
}

void tearDown(void)
{
  // Empty
}

void testFigure8Evaluation(void)
{
  // Fixture
  struct piecewise_traj traj;
  float duration, t;

  traj.t_begin = 2;
  traj.timescale = 1;
  traj.shift = vzero();
  traj.n_pieces = sizeof(figure8_pieces) / sizeof(figure8_pieces[0]);
  traj.pieces = figure8_pieces;
  traj.shift = mkvec(-1, 2, 3);

  // Test
  duration = piecewise_duration(&traj);
#ifdef SHOW_OUTPUT
  printf("t\tx\ty\tz\tyaw\tvx\tvy\tvz\tax\tay\taz\n");
#endif
  for (t = traj.t_begin - 0.5; t < traj.t_begin + duration + 0.5; t += 0.1) {
    struct traj_eval actual = piecewise_eval(&traj, t);
#ifdef SHOW_OUTPUT
    printf(
      "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
      t, actual.pos.x, actual.pos.y, actual.pos.z, actual.yaw,
      actual.vel.x, actual.vel.y, actual.vel.z,
      actual.acc.x, actual.acc.y, actual.acc.z
    );
#endif

    TEST_ASSERT(!isnan(actual.pos.x) && !isnan(actual.pos.y) && !isnan(actual.pos.z));
    TEST_ASSERT(fabs(actual.vel.x) <= 2.5f && fabs(actual.vel.y) <= 2.5f && fabs(actual.vel.z) <= 2.5f);
    TEST_ASSERT(piecewise_is_finished(&traj, t) == (t >= traj.t_begin + duration));
  }

  // Assert
}

void testCompressedFigure8Evaluation(void)
{
  // Fixture
  struct piecewise_traj_compressed traj;
  float duration, t;

  piecewise_compressed_load(&traj, figure8_compressed_pieces);
  traj.t_begin = 2;
  traj.shift = mkvec(-1, 2, 3);

  // Test
  duration = piecewise_compressed_duration(&traj);
#ifdef SHOW_OUTPUT
  printf("t\tx\ty\tz\tyaw\tvx\tvy\tvz\tax\tay\taz\n");
#endif
  for (t = traj.t_begin - 0.5; t < traj.t_begin + duration + 0.5; t += 0.1) {
    struct traj_eval actual = piecewise_compressed_eval(&traj, t);
#ifdef SHOW_OUTPUT
    printf(
      "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
      t, actual.pos.x, actual.pos.y, actual.pos.z, actual.yaw,
      actual.vel.x, actual.vel.y, actual.vel.z,
      actual.acc.x, actual.acc.y, actual.acc.z
    );
#endif

    TEST_ASSERT(!isnan(actual.pos.x) && !isnan(actual.pos.y) && !isnan(actual.pos.z));
    TEST_ASSERT(fabs(actual.vel.x) <= 2.5f && fabs(actual.vel.y) <= 2.5f && fabs(actual.vel.z) <= 2.5f);
    TEST_ASSERT(piecewise_compressed_is_finished(&traj, t) == (t >= traj.t_begin + duration));
  }

  // Assert
}

void testCompressedFrameEvaluation(void)
{
  // Fixture
  struct piecewise_traj_compressed traj;
  float duration, t;

  piecewise_compressed_load(&traj, frame_compressed_pieces);
  traj.t_begin = 0;
  traj.shift = mkvec(0, 0, 0);

  // Test
  duration = piecewise_compressed_duration(&traj);
#ifdef SHOW_OUTPUT
  printf("t\tx\ty\tz\tyaw\tvx\tvy\tvz\tax\tay\taz\n");
#endif
  for (t = traj.t_begin; t < traj.t_begin + duration; t += 0.1) {
    struct traj_eval actual = piecewise_compressed_eval(&traj, t);
#ifdef SHOW_OUTPUT
    printf(
      "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
      t, actual.pos.x, actual.pos.y, actual.pos.z, actual.yaw,
      actual.vel.x, actual.vel.y, actual.vel.z,
      actual.acc.x, actual.acc.y, actual.acc.z
    );
#endif

    TEST_ASSERT(!isnan(actual.pos.x) && !isnan(actual.pos.y) && !isnan(actual.pos.z));
    TEST_ASSERT(fabs(actual.vel.x) <= 2.5f && fabs(actual.vel.y) <= 2.5f && fabs(actual.vel.z) <= 2.5f);
    TEST_ASSERT(piecewise_compressed_is_finished(&traj, t) == (t >= traj.t_begin + duration));
  }

  // Assert
}

void testCompressedSimplifiedFigure8Evaluation(void)
{
  // Fixture
  struct piecewise_traj_compressed traj;
  float duration, t;

  piecewise_compressed_load(&traj, simplified_figure8_compressed_pieces);
  traj.t_begin = 2;
  traj.shift = mkvec(-1, 2, 3);

  // Test
  duration = piecewise_compressed_duration(&traj);
#ifdef SHOW_OUTPUT
  printf("t\tx\ty\tz\tyaw\tvx\tvy\tvz\tax\tay\taz\n");
#endif
  for (t = traj.t_begin - 0.5; t < traj.t_begin + duration + 0.5; t += 0.1) {
    struct traj_eval actual = piecewise_compressed_eval(&traj, t);
#ifdef SHOW_OUTPUT
    printf(
      "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
      t, actual.pos.x, actual.pos.y, actual.pos.z, actual.yaw,
      actual.vel.x, actual.vel.y, actual.vel.z,
      actual.acc.x, actual.acc.y, actual.acc.z
    );
#endif

    TEST_ASSERT(!isnan(actual.pos.x) && !isnan(actual.pos.y) && !isnan(actual.pos.z));
    TEST_ASSERT(piecewise_compressed_is_finished(&traj, t) == (t >= traj.t_begin + duration));
  }

  // Assert
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void testCompressedFigure8RandomOrderQueries(void)
{
  // Fixture
  struct piecewise_traj traj;
  struct piecewise_traj_compressed ctraj;
  float duration, t, diff, maxdiff;
  int i;

  traj.t_begin = 2;
  traj.timescale = 1;
  traj.shift = vzero();
  traj.n_pieces = sizeof(figure8_pieces) / sizeof(figure8_pieces[0]);
  traj.pieces = figure8_pieces;
  traj.shift = mkvec(-1, 2, 3);

  piecewise_compressed_load(&ctraj, figure8_compressed_pieces);
  ctraj.t_begin = 2;
  ctraj.shift = mkvec(-1, 2, 3);

  // Test
  duration = piecewise_compressed_duration(&ctraj);
#ifdef SHOW_OUTPUT
  printf("t\tdiff\tx\ty\tz\tyaw\texp_x\texp_y\texp_z\texp_yaw\tvx\tvy\tvz\tax\tay\taz\n");
#endif
  i = 1000;
  maxdiff = 0.0;
  for (i = 0; i < 10; i++) {
    struct traj_eval actual, expected;

    t = ctraj.t_begin + (rand() / (float)RAND_MAX) * (duration + 1) - 0.5;

    actual = piecewise_compressed_eval(&ctraj, t);
    expected = piecewise_eval(&traj, t);

    diff = 0.0;
    diff = MAX(diff, fabs(actual.pos.x - expected.pos.x));
    diff = MAX(diff, fabs(actual.pos.y - expected.pos.y));
    diff = MAX(diff, fabs(actual.pos.z - expected.pos.z));
    diff = MAX(diff, fabs(actual.yaw - expected.yaw));

#ifdef SHOW_OUTPUT
    printf(
      "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
      t, diff,
      actual.pos.x, actual.pos.y, actual.pos.z, actual.yaw,
      expected.pos.x, expected.pos.y, expected.pos.z, expected.yaw
    );
#endif

    maxdiff = MAX(diff, maxdiff);
  }

  // Assert
  TEST_ASSERT_FLOAT_WITHIN(0.02, 0, maxdiff);
#ifdef SHOW_OUTPUT
  printf("Maximum difference = %.4f\n", maxdiff);
#endif
}
