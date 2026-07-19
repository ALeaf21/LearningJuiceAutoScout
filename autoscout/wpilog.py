"""WPILog binary format writer for AdvantageScope playback.

WPILog is a compact binary format used by FIRST robotics for logging sensor data
and robot poses. AdvantageScope can replay these logs for post-match analysis.

Binary Structure:
1. Magic bytes: "WPILOG" (6 bytes)
2. Version: Major (1 byte), Minor (1 byte)
3. Extra header length (4 bytes)
4. Data records with variable-length encoding:
   - Entry ID (1-8 bytes)
   - Data size (1-8 bytes)
   - Timestamp (1-8 bytes)
   - Payload (variable)

Coordinate Convention (vs AutoScout internal):
- Axes swapped: x_m = internal y_in * 0.0254
- Heading transformed: wpilog_heading = π/2 - internal_heading
- Center-origin: converted from internal corner-origin
"""

import struct


class WPILogWriter:
    """Write robot pose data in WPILog format for AdvantageScope.
    
    Manages variable-length encoding of entry IDs, data sizes, and timestamps
    to minimize file size while supporting arbitrary frame counts and timestamps.
    
    Attributes:
        HEADER_MAGIC (bytes): File magic bytes "WPILOG"
        _fh: Open file handle for binary write
        _next_id: Counter for auto-incrementing entry IDs
    
    Example:
        >>> writer = WPILogWriter("match_log.wpilog")
        >>> pose_eid = writer.start_entry("Robot0/Pose", "double[]")
        >>> vis_eid = writer.start_entry("Robot0/Visible", "boolean")
        >>> writer.write_pose2d(pose_eid, 0, 0.5, -0.3, 0.785)
        >>> writer.write_boolean(vis_eid, 0, True)
        >>> writer.close()
    """
    HEADER_MAGIC = b"WPILOG"

    def __init__(self, path):
        """Initialize WPILog writer and write file header.
        
        Args:
            path (str): Output file path
        """
        self._fh = open(path, "wb")
        self._next_id = 1
        self._fh.write(self.HEADER_MAGIC)
        self._fh.write(struct.pack("<BBI", 1, 0, 0))

    def _encode_int(self, v, max_bytes):
        """Encode integer with minimal bytes (1, 2, 4, or 8).
        
        Args:
            v (int): Integer to encode
            max_bytes (int): Maximum bytes allowed
        
        Returns:
            Tuple[bytes, int]: (encoded_bytes, byte_count)
        """
        for n in [1, 2, 4, 8]:
            if n > max_bytes:
                break
            if v < (1 << (8 * n)):
                return v.to_bytes(n, "little"), n
        return v.to_bytes(max_bytes, "little"), max_bytes

    def _write_record(self, eid, ts, data):
        """Write a single WPILog record with variable-length encoding.
        
        Args:
            eid (int): Entry ID
            ts (int): Timestamp in microseconds
            data (bytes): Record payload
        """
        eb, el = self._encode_int(eid,       4)
        sb, sl = self._encode_int(len(data), 4)
        tb, tl = self._encode_int(ts,        8)
        bf = ((el-1)&3) | (((sl-1)&3)<<2) | (((tl-1)&7)<<4)
        self._fh.write(struct.pack("<B", bf))
        self._fh.write(eb); self._fh.write(sb); self._fh.write(tb)
        self._fh.write(data)

    def start_entry(self, name, type_str):
        """Create a named data stream (e.g., /Robot0/Pose).
        
        Args:
            name (str): Entry name (auto-prefixed with /)
            type_str (str): Data type ("double[]", "boolean", etc.)
        
        Returns:
            int: Entry ID for use with write_pose2d/write_boolean
        """
        if not name.startswith("/"):
            name = "/" + name
        eid = self._next_id; self._next_id += 1
        nb = name.encode(); tb = type_str.encode(); mb = b""
        payload = (struct.pack("<BI", 0, eid) +
                   struct.pack("<I", len(nb)) + nb +
                   struct.pack("<I", len(tb)) + tb +
                   struct.pack("<I", len(mb)) + mb)
        self._write_record(0, 0, payload)
        return eid

    def write_pose2d(self, eid, ts, x_m, y_m, rot_rad):
        """Write a robot Pose2d (x, y, rotation) at a timestamp.
        
        Pose2d is WPILib's standard format for 2D robot positions.
        Coordinates should be in meters using WPILog axis convention.
        
        Args:
            eid (int): Entry ID from start_entry()
            ts (int): Timestamp in microseconds
            x_m (float): Robot X position in meters
            y_m (float): Robot Y position in meters
            rot_rad (float): Robot rotation in radians
        """
        self._write_record(eid, ts,
            struct.pack("<ddd", float(x_m), float(y_m), float(rot_rad)))

    def write_boolean(self, eid, ts, val):
        """Write a boolean value (used for visibility tracking).
        
        Args:
            eid (int): Entry ID from start_entry()
            ts (int): Timestamp in microseconds
            val (bool): Boolean value to write
        """
        self._write_record(eid, ts, struct.pack("<B", 1 if val else 0))

    def close(self):
        """Finalize and close the WPILog file."""
        self._fh.close()
