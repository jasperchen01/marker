from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from PIL import Image

from marker.schema import BlockTypes
from marker.schema.polygon import PolygonBox

if TYPE_CHECKING:
    from marker.schema.document import Document
    from marker.schema.groups.page import PageGroup


class BlockMetadata(BaseModel):
    llm_request_count: int = 0
    llm_error_count: int = 0
    llm_tokens_used: int = 0

    def merge(self, model2):
        return self.__class__(**{
            field: getattr(self, field) + getattr(model2, field)
            for field in self.model_fields
        })


class BlockOutput(BaseModel):
    html: str
    polygon: PolygonBox
    id: BlockId
    children: List[BlockOutput] | None = None
    section_hierarchy: Dict[int, BlockId] | None = None


class BlockId(BaseModel):
    page_id: int
    block_id: Optional[int] = None
    block_type: BlockTypes | None = None

    def __str__(self):
        if self.block_type is None or self.block_id is None:
            return f"/page/{self.page_id}"
        return f"/page/{self.page_id}/{self.block_type.name}/{self.block_id}"

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, (BlockId, str)):
            return NotImplemented

        if isinstance(other, str):
            return str(self) == other
        else:
            return self.page_id == other.page_id and self.block_id == other.block_id and self.block_type == other.block_type

    @field_validator("block_type")
    @classmethod
    def validate_block_type(cls, v):
        from marker.schema import BlockTypes
        if not v in BlockTypes:
            raise ValueError(f"Invalid block type: {v}")
        return v

    def to_path(self):
        return str(self).replace('/', '_')


class Block(BaseModel):
    polygon: PolygonBox
    block_description: str
    block_type: Optional[BlockTypes] = None
    block_id: Optional[int] = None
    page_id: Optional[int] = None
    text_extraction_method: Optional[Literal['pdftext', 'surya', 'gemini']] = None
    structure: List[BlockId] | None = None  # The top-level page structure, which is the block ids in order
    ignore_for_output: bool = False  # Whether this block should be ignored in output
    replace_output_newlines: bool = False  # Whether to replace newlines with spaces in output
    source: Literal['layout', 'heuristics', 'processor'] = 'layout'
    top_k: Optional[Dict[BlockTypes, float]] = None
    metadata: BlockMetadata | None = None
    lowres_image: Image.Image | None = None
    highres_image: Image.Image | None = None
    removed: bool = False # Has block been replaced by new block?

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def id(self) -> BlockId:
        return BlockId(
            page_id=self.page_id,
            block_id=self.block_id,
            block_type=self.block_type
        )

    @classmethod
    def from_block(cls, block: Block) -> Block:
        block_attrs = block.model_dump(exclude=["id", "block_id", "block_type"])
        return cls(**block_attrs)

    def get_image(self, document: Document, highres: bool = False, expansion: Tuple[float, float] | None = None, remove_blocks: Sequence[BlockTypes] | None = None) -> Image.Image | None:
        image = self.highres_image if highres else self.lowres_image
        if image is None:
            page = document.get_page(self.page_id)
            page_image = page.get_image(highres=highres, remove_blocks=remove_blocks)

            # Scale to the image size
            bbox = self.polygon.rescale((page.polygon.width, page.polygon.height), page_image.size)
            if expansion:
                bbox = bbox.expand(*expansion)
            bbox = bbox.bbox
            image = page_image.crop(bbox)
        return image


    def structure_blocks(self, document_page: Document | PageGroup) -> List[Block]:
        if self.structure is None:
            return []
        return [document_page.get_block(block_id) for block_id in self.structure]

    def get_prev_block(self, document_page: Document | PageGroup, block: Block, ignored_block_types: Optional[List[BlockTypes]] = None):
        if ignored_block_types is None:
            ignored_block_types = []

        structure_idx = self.structure.index(block.id)
        if structure_idx == 0:
            return None

        for prev_block_id in reversed(self.structure[:structure_idx]):
            if prev_block_id.block_type not in ignored_block_types:
                return document_page.get_block(prev_block_id)

    def get_next_block(self, document_page: Document | PageGroup, block: Optional[Block] = None, ignored_block_types: Optional[List[BlockTypes]] = None):
        if ignored_block_types is None:
            ignored_block_types = []

        structure_idx = 0
        if block is not None:
            structure_idx = self.structure.index(block.id) + 1

        for next_block_id in self.structure[structure_idx:]:
            if next_block_id.block_type not in ignored_block_types:
                return document_page.get_block(next_block_id)

        return None  # No valid next block found

    def add_structure(self, block: Block):
        if self.structure is None:
            self.structure = [block.id]
        else:
            self.structure.append(block.id)

    def update_structure_item(self, old_id: BlockId, new_id: BlockId):
        if self.structure is not None:
            for i, item in enumerate(self.structure):
                if item == old_id:
                    self.structure[i] = new_id
                    break

    def remove_structure_items(self, block_ids: List[BlockId]):
        if self.structure is not None:
            self.structure = [item for item in self.structure if item not in block_ids]

    def raw_text(self, document: Document) -> str:
        from marker.schema.text.line import Line
        from marker.schema.text.span import Span
        from marker.schema.blocks.tablecell import TableCell

        if self.structure is None:
            if isinstance(self, (Span, TableCell)):
                return self.text
            else:
                return ""

        text = ""
        for block_id in self.structure:
            block = document.get_block(block_id)
            text += block.raw_text(document)
            if isinstance(block, Line) and not text.endswith("\n"):
                text += "\n"
        return text

    def assemble_html(self, document: Document, child_blocks: List[BlockOutput], parent_structure: Optional[List[str]] = None):
        if self.ignore_for_output:
            return ""

        template = ""
        for c in child_blocks:
            template += f"<content-ref src='{c.id}'></content-ref>"

        if self.replace_output_newlines:
            template = template.replace("\n", " ")
            template = "<p>" + template + "</p>"

        return template

    def assign_section_hierarchy(self, section_hierarchy):
        if self.block_type == BlockTypes.SectionHeader and self.heading_level:
            levels = list(section_hierarchy.keys())
            for level in levels:
                if level >= self.heading_level:
                    del section_hierarchy[level]
            section_hierarchy[self.heading_level] = self.id

        return section_hierarchy

    def contained_blocks(self, document: Document, block_types: Sequence[BlockTypes] = None) -> List[Block]:
        if self.structure is None:
            return []

        blocks = []
        for block_id in self.structure:
            block = document.get_block(block_id)
            if block.removed:
                continue
            if (block_types is None or block.block_type in block_types) and not block.removed:
                blocks.append(block)
            blocks += block.contained_blocks(document, block_types)
        return blocks

    def replace_block(self, block: Block, new_block: Block):
        if self.structure is not None:
            for i, item in enumerate(self.structure):
                if item == block.id:
                    self.structure[i] = new_block.id
                    break

    def render(self, document: Document, parent_structure: Optional[List[str]] = None, section_hierarchy: dict | None = None):
        """
        渲染当前块及其子块，并生成 BlockOutput 对象。

        :param document: 文档对象，包含所有块的信息。
        :param parent_structure: 父结构列表，用于传递父块的结构信息。
        :param section_hierarchy: 节层次结构字典，用于跟踪文档的节结构。
        :return: 包含渲染后的 HTML、多边形、ID、子块和节层次结构的 BlockOutput 对象。
        """
        # 存储子块渲染后的内容
        child_content = []
        # 如果没有提供节层次结构，则初始化一个空字典
        if section_hierarchy is None:
            section_hierarchy = {}
        # 分配当前块的节层次结构
        section_hierarchy = self.assign_section_hierarchy(section_hierarchy)

        # 如果当前块有子块结构
        if self.structure is not None and len(self.structure) > 0:
            # 遍历子块的 ID
            for block_id in self.structure:
                # 获取子块对象
                block = document.get_block(block_id)
                # 递归渲染子块
                rendered = block.render(document, self.structure, section_hierarchy)
                # 更新节层次结构，从同级块中获取最新的节层次结构
                section_hierarchy = rendered.section_hierarchy.copy()
                # 将渲染后的子块添加到子内容列表中
                child_content.append(rendered)

        # 返回 BlockOutput 对象，包含渲染后的 HTML、多边形、ID、子块和节层次结构
        return BlockOutput(
            html=self.assemble_html(document, child_content, parent_structure),
            polygon=self.polygon,
            id=self.id,
            children=child_content,
            section_hierarchy=section_hierarchy
        )

    def line_height(self, document: Document) -> float:
        lines = self.contained_blocks(document, (BlockTypes.Line,))
        if len(lines) == 0:
            return 0
        return self.polygon.height / len(lines)

    def update_metadata(self, **kwargs):
        if self.metadata is None:
            self.metadata = BlockMetadata()

        for key, value in kwargs.items():
            metadata_attr = getattr(self.metadata, key)
            if isinstance(metadata_attr, int) and isinstance(value, int):
                setattr(self.metadata, key, metadata_attr + value)
            else:
                raise ValueError(f"Metadata attribute {key} is not an integer")

    def handle_html_output(self, document, child_blocks, parent_structure):
        child_ref_blocks = [block for block in child_blocks if block.id.block_type == BlockTypes.Reference]
        html = Block.assemble_html(self, document, child_ref_blocks, parent_structure)
        return html + self.html
