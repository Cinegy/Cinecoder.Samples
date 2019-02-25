#ifndef __DPX_FILE_H__
#define __DPX_FILE_H__

#define SWAP4(big_e,x) (!(big_e)?(x):uint32_t((((x)>>24)&0xFF)|(((x)>>8)&0xFF00)|(((x)<<8)&0xFF0000)|((x)<<24)))
#define SWAP2(big_e,x) (!(big_e)?(x):uint16_t((((x)>>8)&0xFF)|((x)<<8)))

#pragma pack(push,1)
struct dpx_file_info_t
{
    uint32_t magic_num;
    uint32_t data_offset;
    int8_t   version_str[8];
    uint32_t file_size;
    uint32_t ditto_key;
    uint32_t generic_hdr_size;
    uint32_t industry_hdr_size;
    uint32_t user_data_size;
    int8_t   file_name_str[100];
    int8_t   creation_date_str[24];	// yyyy:mm:dd:hh:mm:ss:LTZ
    int8_t   creator_str[100];
    int8_t   project_str[200];
    int8_t   copyright_str[200];
    uint32_t encryption_key;   		// FFFFFFF = unencrypted
    uint8_t  reserved[104];
};

struct dpx_channel_info_t
{
    uint32_t signage;
    uint32_t ref_low_data;
    int32_t  ref_low_quantity;
    uint32_t ref_high_data;
    int32_t  ref_high_quantity;
    uint8_t  designator;
    uint8_t  transfer_characteristics;
    uint8_t  colourimetry;
    uint8_t  bits_per_pixel;
    uint16_t packing;
    uint16_t encoding;
    uint32_t data_offset;
    uint32_t line_padding;
    uint32_t channel_padding;
    int8_t   description_str[32];
};

struct dpx_image_info_t
{
    uint16_t orientation;
    uint16_t channels_per_image;
    uint32_t pixels_per_line;
    uint32_t lines_per_image;
    dpx_channel_info_t channel[8];
    uint8_t  reserved[52];
};

struct dpx_origin_info_t
{
    uint32_t x_offset;
    uint32_t y_offset;
    int32_t  x_centre;
    int32_t  y_centre;
    uint32_t x_original_size;
    uint32_t y_original_size;
    int8_t   file_name_str[100];
    int8_t   creation_date_str[24];				// yyyy:mm:dd:hh:mm:ss:LTZ
    int8_t   input_device_str[32];
    int8_t   input_serial_number_str[32];
    uint16_t border_validity[4];
    uint32_t pixel_aspect_ratio[2];
    uint8_t  reserved[28];
};

struct dpx_film_info_t
{
    int8_t   film_manufacturer_id[2];
    int8_t   film_type[2];
    int8_t   edge_code_perforation_offset[2];
    int8_t   edge_code_prefix[6];
    int8_t   edge_code_count[4];
    int8_t   film_format_str[32];
    uint32_t frame_position;
    uint32_t sequence_length;
    uint32_t held_count;
    int32_t  frame_rate;
    int32_t  shutter_angle;
    int8_t   frame_identification_str[32];
    int8_t   slate_info_str[100];
    uint8_t  reserved[56];
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
