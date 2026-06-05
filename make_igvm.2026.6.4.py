#!/usr/bin/env python3
"""
IGVM File Generator

This script generates an IGVM (Isolated Guest Virtual Machine) file for a given kernel image.
Currently supports CCA (ARM64 Confidential Compute Architecture) only.
"""

import argparse
import sys
import os
import struct
import re
from typing import Optional, NamedTuple, List, Dict, Any, Set
from abc import ABC, abstractmethod


PAGE_SIZE = 0x1000
ONE_MEGABYTE = 1024 * 1024
IGVM_VHF_PAGE_DATA_UNMEASURED = 0x2


class ARM64KernelHeader(NamedTuple):
    """ARM64 kernel header structure"""
    code0: int          # u32 - Executable code
    code1: int          # u32 - Executable code
    text_offset: int    # u64 - Image load offset, little endian
    image_size: int     # u64 - Effective Image size, little endian
    flags: int          # u64 - kernel flags, little endian
    res2: int           # u64 - reserved
    res3: int           # u64 - reserved
    res4: int           # u64 - reserved
    magic: int          # u32 - Magic number, little endian, "ARM\x64"
    res5: int           # u32 - reserved (used for PE COFF offset)


class VariableHeader(ABC):
    """Abstract base class for IGVM variable headers"""

    def __init__(self, header_type: int):
        """
        Initialize a variable header.

        Args:
            header_type: The IGVM header type constant
        """
        self.header_type = header_type

    @abstractmethod
    def _pack_data(self) -> bytes:
        """Pack the header-specific data into bytes"""
        pass

    def update_file_offset(self, file_offset: int) -> None:
        """Update the file offset in the header data"""
        pass

    def to_bytes(self) -> bytes:
        """
        Convert the variable header to bytes with 8-byte alignment padding.

        Returns:
            Packed binary data for the variable header
        """
        # Pack the data
        data_bytes = self._pack_data()
        data_size = len(data_bytes)

        # Pack the variable header (Type + Length)
        variable_header = struct.pack('<2I', self.header_type, data_size)

        # Combine header and data
        header = variable_header + data_bytes

        # Pad to 8-byte alignment
        padding_needed = (8 - (len(header) % 8)) % 8
        if padding_needed > 0:
            header += b'\x00' * padding_needed

        return header

    def __len__(self) -> int:
        """Return the length of the packed header"""
        return len(self.to_bytes())

    def data_size(self) -> int:
        """Return the data size"""
        return 0

class SupportedPlatformHeader(VariableHeader):
    """IGVM_VHS_SUPPORTED_PLATFORM variable header"""

    def __init__(self):
        super().__init__(0x1)           # IGVM_VHT_SUPPORTED_PLATFORM = 0x1
        self.compatibility_mask = 0x1
        self.highest_vtl = 0x0
        self.platform_type = 0x6        # IgvmPlatformRme
        self.platform_version = 0x1     # IGVM_RME_PLATFORM_VERSION
        self.shared_gpa_boundary = 0    # Shared boundary is fixed at (IPA width - 1)

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_SUPPORTED_PLATFORM structure"""
        return struct.pack('<I2BHQ',
            self.compatibility_mask,
            self.highest_vtl,
            self.platform_type,
            self.platform_version,
            self.shared_gpa_boundary
        )


class ParameterHeader(VariableHeader):
    """IGVM_VHS_PARAMETER variable header"""

    def __init__(self, paramType: int, parameter_area_index: int = 0, byte_offset: int = 0):
        super().__init__(paramType)
        self.parameter_area_index = parameter_area_index
        self.byte_offset = byte_offset

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_PARAMETER structure"""
        return struct.pack('<2I',
            self.parameter_area_index,
            self.byte_offset
        )


class ParameterAreaHeader(VariableHeader):
    """IGVM_VHS_PARAMETER_AREA variable header"""

    def __init__(self, number_of_bytes: int, parameter_area_index: int):
        super().__init__(0x301)         # IGVM_VHT_PARAMETER_AREA = 0x301
        self.number_of_bytes = number_of_bytes
        self.parameter_area_index = parameter_area_index
        self.file_offset = 0

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_PARAMETER_AREA structure"""
        return struct.pack('<Q2I',
            self.number_of_bytes,
            self.parameter_area_index,
            self.file_offset
        )

    def update_file_offset(self, file_offset: int) -> None:
        """Update the file offset in the parameter area header"""
        self.file_offset = file_offset


class ParameterInsertHeader(VariableHeader):
    """IGVM_VHS_PARAMETER_INSERT variable header"""

    def __init__(self, gpa: int, parameter_area_index: int):
        super().__init__(0x303)  # IGVM_VHT_PARAMETER_INSERT = 0x303
        self.gpa = gpa
        self.compatibility_mask = 0x1
        self.parameter_area_index = parameter_area_index

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_PARAMETER_INSERT structure"""
        return struct.pack('<Q2I',
            self.gpa,
            self.compatibility_mask,
            self.parameter_area_index
        )


class VpContextHeader(VariableHeader):
    """IGVM_VHS_VP_CONTEXT variable header"""

    def __init__(self, context_data_size: int):
        super().__init__(0x304)  # IGVM_VHT_VP_CONTEXT = 0x304
        self.context_gpa = 0x0
        self.compatibility_mask = 0x1
        self.file_offset = 0
        self.vp_index = 0
        self.context_data_size = context_data_size

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_VP_CONTEXT structure"""
        return struct.pack('<Q2I2H',
            self.context_gpa,
            self.compatibility_mask,
            self.file_offset,
            self.vp_index,
            0
        )

    def update_file_offset(self, file_offset: int) -> None:
        """Update the file offset in the VP context header"""
        self.file_offset = file_offset

    def data_size(self) -> int:
        """Return the data size"""
        return self.context_data_size

class PageDataHeader(VariableHeader):
    """IGVM_VHS_PAGE_DATA variable header"""

    def __init__(self, gpa: int, flags: int = 0x0, data_type: int = 0x00, has_file_data: bool = True):
        super().__init__(0x302)  # IGVM_VHT_PAGE_DATA = 0x302
        self.gpa = gpa
        self.compatibility_mask = 0x1
        self.file_offset = 0
        self.flags = flags
        self.data_type = data_type
        self.has_file_data = has_file_data

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_PAGE_DATA structure"""
        return struct.pack('<Q3I2H',
            self.gpa,
            self.compatibility_mask,
            self.file_offset,
            self.flags,
            self.data_type,
            0
        )

    def update_file_offset(self, file_offset: int) -> None:
        """Update the file offset in the page data header"""
        if self.has_file_data:
            self.file_offset = file_offset
        else:
            self.file_offset = 0

    def data_size(self) -> int:
        """Return the data size"""
        if self.has_file_data:
            return PAGE_SIZE

        return 0


class CvmPolicyHeader(VariableHeader):
    """IGVM_VHS_CVM_POLICY variable header"""

    def __init__(self, policy: int):
        """
        Initialize a CVM policy header.

        Args:
            policy: 64-bit policy value (e.g., HV_RME_GUEST_POLICY)
        """
        super().__init__(0x101)  # IGVM_VHT_CVM_POLICY = 0x101
        self.policy = policy
        self.compatibility_mask = 0x1
        self.policy = policy

    def _pack_data(self) -> bytes:
        """Pack the IGVM_VHS_CVM_POLICY structure"""
        return struct.pack('<Q2I',
            self.policy,
            self.compatibility_mask,
            0
        )


def build_rme_guest_policy(hash_algorithm: str, debug_allowed: bool = False) -> int:
    """
    Build HV_RME_GUEST_POLICY value from hash algorithm and debug settings.

    Args:
        hash_algorithm: Hash algorithm string (SHA-256, SHA-512, or SHA-384)
        debug_allowed: Whether debugging is allowed for the Realm

    Returns:
        64-bit policy value for HV_RME_GUEST_POLICY
    """
    # Map hash algorithm strings to values according to HV_RME_GUEST_POLICY
    # 0 - SHA-256
    # 1 - SHA-512
    # 2 - SHA-384
    hash_algorithm_map = {
        "SHA-256": 0,
        "SHA-512": 1,
        "SHA-384": 2
    }

    # Build the policy value according to HV_RME_GUEST_POLICY structure
    # Bit 0: DebugAllowed
    # Bits 1-8: HashAlgorithm (8 bits)
    # Bits 9-63: Reserved (55 bits)
    policy = 0

    if debug_allowed:
        policy |= (1 << 0)  # Set DebugAllowed bit

    hash_value = hash_algorithm_map.get(hash_algorithm, 0)
    policy |= (hash_value << 1)  # Set HashAlgorithm bits (1-8)

    return policy


def read_arm64_kernel_header(kernel_image_path: str) -> Optional[ARM64KernelHeader]:
    """
    Read the ARM64 kernel header from the kernel image file.

    Args:
        kernel_image_path: Path to the kernel image file

    Returns:
        ARM64KernelHeader object or None if failed
    """
    try:
        with open(kernel_image_path, 'rb') as f:
            # Read the first 64 bytes (ARM64 kernel header size)
            header_data = f.read(64)

            if len(header_data) < 64:
                print(f"Error: Kernel image too small ({len(header_data)} bytes), expected at least 64 bytes")
                return None

            # Unpack the header structure
            # Format: 2 UINT32 + 6 UINT64 + 2 UINT32 = 8 + 48 + 8 = 64 bytes
            unpacked = struct.unpack('<2I6Q2I', header_data)

            header = ARM64KernelHeader(
                code0=unpacked[0],
                code1=unpacked[1],
                text_offset=unpacked[2],
                image_size=unpacked[3],
                flags=unpacked[4],
                res2=unpacked[5],
                res3=unpacked[6],
                res4=unpacked[7],
                magic=unpacked[8],
                res5=unpacked[9]
            )

            # Validate magic number
            expected_magic = 0x644d5241  # "ARM\x64"
            if header.magic != expected_magic:
                magic_bytes = struct.pack('<I', header.magic)
                magic_str = magic_bytes.decode('ascii', errors='replace')
                raise ValueError(f"Invalid ARM64 kernel magic: 0x{header.magic:08x} ('{magic_str}'), expected 0x{expected_magic:08x} ('ARM\\x64')")

            return header

    except Exception as e:
        print(f"Error reading kernel header: {e}")
        return None


def dump_arm64_kernel_header(header: ARM64KernelHeader) -> None:
    """
    Dump the ARM64 kernel header to console.

    Args:
        header: ARM64KernelHeader object to dump
    """
    print("\n=== ARM64 Kernel Header ===")
    print(f"code0:       0x{header.code0:08x}")
    print(f"code1:       0x{header.code1:08x}")
    print(f"text_offset: 0x{header.text_offset:016x} ({header.text_offset} bytes)")
    print(f"image_size:  0x{header.image_size:016x} ({header.image_size} bytes)")
    print(f"flags:       0x{header.flags:016x}")

    magic_bytes = struct.pack('<I', header.magic)
    magic_str = magic_bytes.decode('ascii', errors='replace')
    print(f"magic:       0x{header.magic:08x} ('{magic_str}')")

    print("=" * 30)


def calculate_kernel_base_gpa(kernel_header: ARM64KernelHeader) -> int:
    """
    Calculate the kernel base GPA - 2MB aligned address above 32MB plus text_offset.

    Args:
        kernel_header: ARM64 kernel header containing text_offset

    Returns:
        Base GPA for kernel loading
    """
    # Start above 32MB
    base_address = 32 * 1024 * 1024  # 32MB = 0x2000000

    # Align to 2MB boundary
    alignment = 2 * 1024 * 1024  # 2MB = 0x200000
    kernel_base = (base_address + alignment - 1) & ~(alignment - 1)

    # Add text_offset from kernel header
    kernel_base += kernel_header.text_offset

    return kernel_base


def create_page_data_headers(kernel_size: int, kernel_base_gpa: int) -> List[PageDataHeader]:
    """
    Create IGVM_VHS_PAGE_DATA headers for every 4KB page in the kernel image.

    Args:
        kernel_size: Size of the kernel image in bytes
        kernel_base_gpa: Base GPA where kernel should be loaded

    Returns:
        List of PageDataHeader objects
    """
    # Calculate number of pages needed
    num_pages = (kernel_size + PAGE_SIZE - 1) // PAGE_SIZE

    page_headers = []
    for page_idx in range(num_pages):
        # Calculate GPA for this page
        page_gpa = kernel_base_gpa + (page_idx * PAGE_SIZE)

        # Create page data header
        page_header = PageDataHeader(gpa=page_gpa)
        page_headers.append(page_header)

    return page_headers


def create_default_accept_page_data_headers(variable_headers: List[VariableHeader], accept_memory_range_mb: int) -> List[PageDataHeader]:
    """
    Create IGVM_VHS_PAGE_DATA headers (with no file data) for uncovered pages.

    Args:
        variable_headers: Existing list of variable headers
        accept_memory_range_mb: Number of MB of RAM to accept by default

    Returns:
        List of PageDataHeader objects for uncovered pages in [0, accept_memory_range_mb)
    """
    if accept_memory_range_mb <= 0:
        return []

    covered_pages: Set[int] = set()
    parameter_area_sizes: Dict[int, int] = {}

    for header in variable_headers:
        if isinstance(header, PageDataHeader):
            covered_pages.add(header.gpa // PAGE_SIZE)
        elif isinstance(header, ParameterAreaHeader):
            parameter_area_sizes[header.parameter_area_index] = header.number_of_bytes

    for header in variable_headers:
        if isinstance(header, ParameterInsertHeader):
            parameter_area_size = parameter_area_sizes.get(header.parameter_area_index, PAGE_SIZE)
            page_count = (parameter_area_size + PAGE_SIZE - 1) // PAGE_SIZE
            start_page = header.gpa // PAGE_SIZE

            for page_index in range(page_count):
                covered_pages.add(start_page + page_index)

    accepted_bytes = accept_memory_range_mb * ONE_MEGABYTE
    accepted_page_count = accepted_bytes // PAGE_SIZE

    default_accept_headers: List[PageDataHeader] = []
    for page_number in range(accepted_page_count):
        if page_number not in covered_pages:
            default_accept_headers.append(
                PageDataHeader(
                    gpa=page_number * PAGE_SIZE,
                    flags=IGVM_VHF_PAGE_DATA_UNMEASURED,
                    has_file_data=False
                )
            )

    return default_accept_headers


def validate_architecture(arch: str) -> bool:
    """
    Validate the architecture parameter.

    Args:
        arch: The architecture string to validate

    Returns:
        True if valid, False otherwise
    """
    allowed_architectures = ["CCA"]
    return arch.upper() in allowed_architectures


def generate_igvm_file(
    kernel_image_path: str,
    architecture: str,
    output_path: Optional[str] = None,
    hash_algorithm: Optional[str] = None,
    accept_memory_range_mb: Optional[int] = None
) -> bool:
    """
    Generate an IGVM file for the given kernel image.

    Args:
        kernel_image_path: Path to the kernel image file
        architecture: Target architecture (currently only "CCA" is supported)
        output_path: Optional output path for the IGVM file
        hash_algorithm: Optional hash algorithm for Realm measurement (SHA-256, SHA-512, SHA-384)
        accept_memory_range_mb: Optional number of MB of RAM accepted by default

    Returns:
        True if successful, False otherwise
    """
    print(f"Generating IGVM file for architecture: {architecture}")
    print(f"Kernel image: {kernel_image_path}")

    if output_path:
        print(f"Output file: {output_path}")
    else:
        # Generate default output filename
        base_name = os.path.basename(kernel_image_path)
        output_path = f"{base_name}.bin"
        print(f"Output file: {output_path}")

    if hash_algorithm:
        print(f"Hash algorithm: {hash_algorithm}")

    if accept_memory_range_mb is not None:
        print(f"Accept memory range: {accept_memory_range_mb} MB")

    if architecture.upper() == "CCA":
        return generate_cca_igvm(kernel_image_path, output_path, hash_algorithm, accept_memory_range_mb)

    return False


def create_igvm_fixed_header_v2(variable_header_size: int, total_file_size: int) -> bytes:
    """
    Create IGVM_FIXED_HEADER_V2 structure.

    Args:
        variable_header_size: Size of the variable header section
        total_file_size: Total size of the IGVM file

    Returns:
        Packed binary data for the fixed header
    """
    # IGVM constants from IgvmFileFormat.h
    IGVM_MAGIC_VALUE = 0x4D564749  # "IGVM"
    IGVM_REV_V2 = 2
    IGVM_ARCH_AARCH64 = 0x1  # CCA uses AArch64

    # Fixed header V2 is 0x20 bytes (32 bytes)
    fixed_header_size = 0x20
    variable_header_offset = fixed_header_size

    # Page size for AArch64 CCA
    page_size = 0x1000  # 4KB

    # Calculate checksum placeholder (will be updated later)
    checksum = 0  # TODO: Calculate actual checksum

    # Pack the IGVM_FIXED_HEADER_V2 structure
    # struct format: 8 UINT32 values (Little-endian)
    header_data = struct.pack('<8I',
        IGVM_MAGIC_VALUE,          # Magic
        IGVM_REV_V2,               # FormatVersion
        variable_header_offset,    # VariableHeaderOffset
        variable_header_size,      # VariableHeaderSize
        total_file_size,           # TotalFileSize
        checksum,                  # Checksum (placeholder)
        IGVM_ARCH_AARCH64,         # Arch
        page_size                  # PageSize
    )

    return header_data


def generate_cca_igvm(
    kernel_image_path: str,
    output_path: str,
    hash_algorithm: Optional[str] = None,
    accept_memory_range_mb: Optional[int] = None
) -> bool:
    """
    Generate IGVM file for CCA (Confidential Compute Architecture).

    Args:
        kernel_image_path: Path to the kernel image file
        output_path: Path where the IGVM file will be created
        hash_algorithm: Optional hash algorithm for Realm measurement (SHA-256, SHA-512, SHA-384)
        accept_memory_range_mb: Optional number of MB of RAM accepted by default

    Returns:
        True if successful, False otherwise
    """
    try:
        print("Generating CCA IGVM file...")

        # Get kernel image size
        kernel_size = os.path.getsize(kernel_image_path)
        print(f"Kernel image size: {kernel_size} bytes")

        # Read and dump the ARM64 kernel header
        kernel_header = read_arm64_kernel_header(kernel_image_path)
        if kernel_header is None:
            print("Failed to read kernel header")
            return False

        dump_arm64_kernel_header(kernel_header)

        # Calculate kernel base GPA and DT base GPA
        dt_base_address = 24 * 1024 * 1024  # 24 MB
        kernel_base_gpa = calculate_kernel_base_gpa(kernel_header)
        print(f"Kernel base GPA: 0x{kernel_base_gpa:016x}, DT base GPA: 0x{dt_base_address:016x}")

        # Create VP context data
        x_regs = [0] * 8                # X0-X7 registers
        x_regs[0] = dt_base_address     # X0 points to DT
        pc = kernel_base_gpa            # Program counter points to kernel entry point
        vp_context_data = struct.pack('<8QQ', *x_regs, pc)

        # Add headers in the order they should appear in the file
        variable_headers = []
        variable_headers.append(SupportedPlatformHeader())

        # Add CVM policy header if hash algorithm is specified
        if hash_algorithm:
            policy = build_rme_guest_policy(hash_algorithm)
            variable_headers.append(CvmPolicyHeader(policy=policy))

        variable_headers.append(VpContextHeader(len(vp_context_data)))

        # Create a parameter area for device tree (2 MB)
        IGVM_VHT_DEVICE_TREE = 0x312
        variable_headers.append(ParameterAreaHeader(number_of_bytes= 2 * 1024 * 1024, parameter_area_index = 0))
        variable_headers.append(ParameterHeader(IGVM_VHT_DEVICE_TREE, parameter_area_index = 0, byte_offset = 0x0))
        variable_headers.append(ParameterInsertHeader(gpa = dt_base_address, parameter_area_index = 0))

        page_data_headers = create_page_data_headers(kernel_size, kernel_base_gpa)
        variable_headers.extend(page_data_headers)

        # Add default-accepted page data headers for uncovered pages if accept_memory_range_mb is specified
        if accept_memory_range_mb is not None and accept_memory_range_mb > 0:
            default_accept_headers = create_default_accept_page_data_headers(variable_headers, accept_memory_range_mb)
            variable_headers.extend(default_accept_headers)
            print(f"Added default-accepted pages: {len(default_accept_headers)}")

        # Calculate file sizes and offsets
        fixed_header_size = 0x20  # IGVM_FIXED_HEADER_V2 size
        variable_header_size = sum(len(header) for header in variable_headers)

        # Update headers with correct file offsets
        current_data_offset = fixed_header_size + variable_header_size
        for header in variable_headers:
            header.update_file_offset(current_data_offset)
            current_data_offset += header.data_size()

        # Create the fixed header
        fixed_header = create_igvm_fixed_header_v2(variable_header_size, current_data_offset)

        # Write the IGVM file
        with open(output_path, 'wb') as f:
            # Write fixed header
            f.write(fixed_header)

            # Write all variable headers
            for header in variable_headers:
                f.write(header.to_bytes())

            # Write VP context data
            f.write(vp_context_data)

            # Write kernel data (page by page, padded to 4KB)
            with open(kernel_image_path, 'rb') as kernel_file:
                bytes_written = 0
                page_num = 0
                while bytes_written < kernel_size:
                    # Read up to one page
                    page_data = kernel_file.read(min(PAGE_SIZE, kernel_size - bytes_written))

                    # Pad to full page size if needed
                    if len(page_data) < PAGE_SIZE:
                        page_data += b'\x00' * (PAGE_SIZE - len(page_data))

                    f.write(page_data)
                    bytes_written += len(page_data)
                    page_num += 1

                print(f"Written kernel data: {page_num} pages, {bytes_written} bytes")

        print(f"IGVM file created: {output_path}")
        return True

    except Exception as e:
        print(f"Error generating CCA IGVM file: {e}")
        return False


def main():
    """Main entry point for the IGVM generator."""

    def parse_accept_memory_range(value: str) -> int:
        """Parse an accept memory range value in the format <number>MB."""
        match = re.fullmatch(r'(\d+)MB', value)
        if match is None:
            raise argparse.ArgumentTypeError("accept-memory-range must be specified as <number>MB, for example 512MB")

        return int(match.group(1))

    parser = argparse.ArgumentParser(
        description="Generate an IGVM file for a given kernel image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_igvm.py --architecture CCA --kernel-image Image
  python make_igvm.py --architecture CCA --kernel-image Image --output mykernel.bin
  python make_igvm.py --architecture CCA --kernel-image Image --accept-memory-range 512MB
        """
    )

    parser.add_argument(
        "--architecture",
        required=True,
        help="Target architecture (currently only 'CCA' is supported, case insensitive)"
    )

    parser.add_argument(
        "--kernel-image",
        required=True,
        help="Path to the kernel image file. This must be the default flat binary image."
    )

    parser.add_argument(
        "--output",
        help="Output path for the IGVM file (optional)"
    )

    parser.add_argument(
        "--hash-algorithm",
        choices=["SHA-256", "SHA-512", "SHA-384"],
        help="Hash algorithm used to measure the initial state of the Realm (optional)"
    )

    parser.add_argument(
        "--accept-memory-range",
        type=parse_accept_memory_range,
        help="Amount of RAM to accept by default, specified as <number>MB (optional), e.g. 512MB"
    )

    args = parser.parse_args()

    # Validate architecture
    if not validate_architecture(args.architecture):
        print(f"Error: Unsupported architecture '{args.architecture}'")
        print("Supported architectures: CCA (case insensitive)")
        sys.exit(1)

    # Validate kernel image path
    if not os.path.exists(args.kernel_image):
        print(f"Error: Kernel image file not found: {args.kernel_image}")
        sys.exit(1)

    # Generate IGVM file
    success = generate_igvm_file(
        args.kernel_image,
        args.architecture,
        args.output,
        args.hash_algorithm,
        args.accept_memory_range
    )

    if success:
        print("IGVM file generation completed successfully")
        sys.exit(0)
    else:
        print("IGVM file generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
