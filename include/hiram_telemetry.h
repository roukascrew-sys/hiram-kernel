#ifndef HIRAM_TELEMETRY_H
#define HIRAM_TELEMETRY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define HIRAM_TELEM_MAGIC 0x48524D31UL /* "HRM1" */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t sequence_id;
    int8_t   lidar_obs;
    int8_t   tof_obs;
    int8_t   slip_obs;
    uint8_t  action;
    uint8_t  fault_flags;
    uint8_t  reserved[3];
    uint32_t cycle_count;
    uint32_t payload_crc32;
} hiram_telem_packet_t;
#pragma pack(pop)

#define HIRAM_TELEM_RAW_LEN (sizeof(hiram_telem_packet_t))
/* COBS overhead: raw_len + ceil(raw_len / 254) + 1 + 1 delimiter */
#define HIRAM_TELEM_ENCODED_MAX (HIRAM_TELEM_RAW_LEN + 4)

size_t hiram_telem_pack_and_encode(
    const hiram_telem_packet_t *pkt,
    uint8_t *out_buf,
    size_t out_buf_max
);

#endif /* HIRAM_TELEMETRY_H */
