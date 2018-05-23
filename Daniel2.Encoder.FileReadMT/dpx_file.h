#ifndef __DPX_FILE_H__
#define __DPX_FILE_H__

#define SWAP4(big_e,x) (!(big_e)?(x):DWORD((((x)>>24)&0xFF)|(((x)>>8)&0xFF00)|(((x)<<8)&0xFF0000)|((x)<<24)))
#define SWAP2(big_e,x) (!(big_e)?(x):WORD((((x)>>8)&0xFF)|((x)<<8)))

#pragma pack(push,1)
struct dpx_file_info_t
{
    UINT   magic_num;
    UINT   data_offset;
    CHAR   version_str[8];
    UINT   file_size;
    UINT   ditto_key;
    UINT   generic_hdr_size;
    UINT   industry_hdr_size;
    UINT   user_data_size;
    CHAR   file_name_str[100];
    CHAR   creation_date_str[24];	// yyyy:mm:dd:hh:mm:ss:LTZ
    CHAR   creator_str[100];
    CHAR   project_str[200];
    CHAR   copyright_str[200];
    UINT   encryption_key;   		// FFFFFFF = unencrypted
    CHAR   reserved[104];
};

struct dpx_channel_info_t
{
    UINT   signage;
    UINT   ref_low_data;
    INT    ref_low_quantity;
    UINT   ref_high_data;
    INT    ref_high_quantity;
    UCHAR  designator;
    UCHAR  transfer_characteristics;
    UCHAR  colourimetry;
    UCHAR  bits_per_pixel;
    WORD   packing;
    WORD   encoding;
    UINT   data_offset;
    UINT   line_padding;
    UINT   channel_padding;
    CHAR   description_str[32];
};

struct dpx_image_info_t
{
    WORD   orientation;
    WORD   channels_per_image;
    UINT   pixels_per_line;
    UINT   lines_per_image;
    dpx_channel_info_t channel[8];
    CHAR   reserved[52];
};

struct dpx_origin_info_t
{
    UINT   x_offset;
    UINT   y_offset;
    INT    x_centre;
    INT    y_centre;
    UINT   x_original_size;
    UINT   y_original_size;
    CHAR   file_name_str[100];
    CHAR   creation_date_str[24];				// yyyy:mm:dd:hh:mm:ss:LTZ
    CHAR   input_device_str[32];
    CHAR   input_serial_number_str[32];
    WORD   border_validity[4];
    UINT   pixel_aspect_ratio[2];
    CHAR   reserved[28];
};

struct dpx_film_info_t
{
    CHAR   film_manufacturer_id[2];
    CHAR   film_type[2];
    CHAR   edge_code_perforation_offset[2];
    CHAR   edge_code_prefix[6];
    CHAR   edge_code_count[4];
    CHAR   film_format_str[32];
    UINT   frame_position;
    UINT   sequence_length;
    UINT   held_count;
    INT    frame_rate;
    INT    shutter_angle;
    CHAR   frame_identification_str[32];
    CHAR   slate_info_str[100];
    CHAR   reserved[56];
};

struct dpx_file_header_t
{
    dpx_file_info_t   file;
    dpx_image_info_t  image;
    dpx_origin_info_t origin;
    dpx_film_info_t   film;
};
#pragma pack(pop)

#endif
