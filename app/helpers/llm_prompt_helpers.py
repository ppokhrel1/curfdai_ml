
from typing import Dict, Any, List
import json
import yaml
from sqlalchemy import create_engine, text
import uuid

from app.configs.config import config
from app.db.engine import DatabaseManager

engine = DatabaseManager(config.DATABASE_URL).engine

async def _analyze_requirements_with_llm(llm_client, user_prompt: str) -> Dict[str, Any]:
    """Use Qwen to analyze requirements with simplified prompt"""
    
    prompt = f"""
    <|im_start|>system
    You are a mechanical engineering expert. Extract technical requirements from descriptions.
    Return ONLY valid JSON, no other text. No markdown, no code fences, no explanations.<|im_end|>
    <|im_start|>user
    Extract technical requirements from this description: "{user_prompt}"

    Return JSON with these fields:
    - model_type (robot, automobile, drone, human, aircraft, flying_wing, or custom)
    - primary_function (main purpose)
    - key_components (list of critical parts)
    - mobility_type (wheeled, legged, flying, driving, fixed, etc)
    - environment (indoor, outdoor, aerial, underwater, etc)
    - size_constraints
    - performance_requirements (with speed, payload, endurance)
    - target_simulator
    - complexity_level (simple, moderate, complex)

    Make technically sound assumptions for missing information.
    -- Provide a working implementation that can be loaded in editor
    Return ONLY the JSON object.<|im_end|>
    """
    
    response = await llm_client.generate_json(prompt)  # Remove the empty string
    
    # Validate the response has required fields
    required_fields = ['model_type', 'primary_function', 'key_components', 'mobility_type']
    for field in required_fields:
        if field not in response:
            response[field] = "unknown"
    
    return response


async def _create_assembly_plan_with_llm(llm_client, requirements: Dict[str, Any], available_parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use Qwen to create assembly plan with improved prompt for structured assemblies."""
    
    # Handle empty parts list gracefully
    parts_context = "No specific parts available - create a generic design."
    if available_parts:
        # Include mesh information for the LLM to select from
        parts_context = "\n".join([
            f"- {part['name']} ({part['category']}): {part.get('description', 'No description')} [MESH: {part.get('mesh_file', 'NONE')}.{part.get('mesh_format', 'none')}]"
            for part in available_parts[:15]
        ])
    
    # Simplify the prompt structure
    prompt = f"""
    <|im_start|>system
    You are a mechanical design engineer. Create practical assembly plans.
    Return ONLY valid JSON, no other text. No markdown, no code fences.<|im_end|>
    <|im_start|>user
    Create an assembly plan for: {requirements.get('primary_function', 'mechanical system')}

    Requirements: {json.dumps(requirements, indent=2)}

    Available parts: {parts_context}

    CRITICAL STRUCTURAL INSTRUCTIONS FOR ANY GADGET:
    1. **Distinct Parts:** If the design requires 'N' instances of a component (e.g., 4 wheels, 2 wings, 6 legs), you MUST create 'N' distinct entries in the "parts" list (e.g., wheel_1, wheel_2, ..., wheel_N). DO NOT use a single part entry with a list of placements.
    2. **Base Link:** Name the primary structural link the "base_link" and place it at (0, 0, 0).
    3. **Joint Types:** Use **fixed** joints for all non-moving components attached to a parent (e.g., batteries, sensors, frames). Use **continuous**, **revolute**, or **prismatic** only for motion (e.g., wheels, rotors, robotic joints).
    4. **Placement:** All 'placement' values must be calculated relative to the **parent** link's origin, ensuring no major components overlap unless structurally necessary.

    Create a realistic design with these constraints. Return ONLY JSON with this structure:
    {{
        "model_name": "descriptive_name",
        "model_type": "type_from_requirements", 
        "description": "brief_description",
        "parts": [
            {{
                "name": "part_name",
                "category": "part_category", 
                "role": "purpose_in_assembly",
                "placement": {{"x": <relative_x>, "y": <relative_y>, "z": <relative_z>}},
                "mesh_file": "file_id_or_path.stl", 
                "mesh_format": "stl",
                "mass": <relative_mass>,
                "dimensions": {{"length": <relative_length>, "width": <relative_width>, "height": <relative_height>}}
            }}
        ],
        "joints": [
            {{
                "name": "joint_name",
                "type": "fixed|revolute|continuous|prismatic", 
                "parent": "parent_part_name",
                "child": "child_part_name", 
                "axis": "rotation_axis_if_applicable"
            }}
        ],
        "parameters": {{
            "mass": <relative_mass>,
            "dimensions": {{"length": <relative_length>, "width": <relative_width>, "height": <relative_height>}},
            "performance_specs": {{
                "main_purpose": "value",
                "speed": "value", 
                "payload": "value",
                "endurance": "value"
            }}
        }},
        "assembly_notes": "special_instructions"
    }}
    -- Provide a working implementation that can be loaded in editor

    Use realistic values and ensure the design matches the requirements. Make sure to choose a mesh_file and mesh_format from the available parts for each component. Apply the CRITICAL STRUCTURAL INSTRUCTIONS rigorously.<|im_end|>
    """
    
    return await llm_client.generate_json(prompt)  # Remove the empty string


async def _generate_yaml_content(llm_client, assembly_plan: Dict[str, Any]) -> str:
    """Generate YAML content that can be parsed to SDF format later"""
    
    prompt = f"""
    <|im_start|>system
    You are a Gazebo SDF expert. Generate a YAML representation of a robot model that can be converted to SDF format.
    The YAML should represent the complete model structure including links, joints, and plugins.
    Return ONLY valid YAML content, no explanations, no markdown, no code fences.<|im_end|>
    <|im_start|>user

    IMPORTANT: Every link MUST have an inertial section with mass and inertia values.
    Use appropriate mass values based on part category:
    - Base/chassis: 1.0-5.0 kg
    - Wheels/motors: 0.5-1.0 kg  
    - Sensors/cameras: 0.1-0.3 kg
    - Batteries: 0.5-2.0 kg
    - Small components: 0.05-0.2 kg
    
    Create a YAML representation for this robot model that can be converted to SDF:

    Model Name: {assembly_plan['model_name']}
    Description: {assembly_plan.get('description', 'Automatically generated model')}
    
    Parts/Links:
    {_format_links_for_yaml(assembly_plan)}
    
    Joints:
    {_format_joints_for_yaml(assembly_plan)}

    Generate YAML with this structure:
    
    model:
    name: "{assembly_plan['model_name']}"
    version: "1.0"
    description: "{assembly_plan.get('description', 'Automatically generated model')}"
    links:
        - name: "base_link"
        pose: [0, 0, 0, 0, 0, 0]
        visual:
            geometry:
            type: "mesh"
            uri: "package://model_library/meshes/base.stl"
        collision:
            geometry:
            type: "mesh" 
            uri: "package://model_library/meshes/base.stl"
        inertial:
            mass: 1.0
            inertia: [0.1, 0, 0, 0.1, 0, 0.1]
    joints:
        - name: "wheel_joint"
        type: "revolute"
        parent: "base_link"
        child: "wheel"
        axis: [0, 0, 1]
        pose: [0.2, 0, 0, 0, 0, 0]
    plugins:
        - name: "gazebo_ros_control"
        filename: "libgazebo_ros_control.so"
        parameters:
            robotNamespace: "/{assembly_plan['model_name']}"

    Use realistic values based on the model description and ensure all required fields are present.
    Return ONLY the YAML content.<|im_end|>
    """
    
    response = await llm_client.generate_text(
        prompt,
        temperature=0.1,
        max_tokens=3000
    )
    
    return _clean_yaml_response(response)


def _format_links_for_yaml(assembly_plan: Dict[str, Any]) -> str:
    """Format links information for YAML generation"""
    links_info = []
    for part in assembly_plan.get('parts', []):
        link_info = f"    - {part['name']}: category={part.get('category', 'unknown')}, "
        if 'placement' in part:
            pos = part['placement']
            if isinstance(pos, dict):
                link_info += f"position=({pos.get('x', 0)}, {pos.get('y', 0)}, {pos.get('z', 0)})"
            elif isinstance(pos, list):
                links_info += ",".join(f"position=({val.get('x', 0)}, {val.get('y', 0)}, {val.get('z', 0)})" for val in pos)
        if 'mesh_file' in part:
            link_info += f", mesh={part['mesh_file']}.{part.get('mesh_format', 'stl')}"
        links_info.append(link_info)
    
    return '\n'.join(links_info) if links_info else "    No parts defined"

def _format_joints_for_yaml(assembly_plan: Dict[str, Any]) -> str:
    """Format joints information for YAML generation"""
    joints_info = []
    for joint in assembly_plan.get('joints', []):
        joint_info = f"    - {joint['name']}: type={joint['type']}, parent={joint['parent']}, child={joint['child']}"
        if 'axis' in joint:
            joint_info += f", axis={joint['axis']}"
        joints_info.append(joint_info)
    
    return '\n'.join(joints_info) if joints_info else "    No joints defined"

def _clean_yaml_response(response: str) -> str:
    """Clean YAML response and ensure it's valid"""
    cleaned = response.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    cleaned = cleaned.replace('```yaml', '').replace('```', '').strip()
    
    # Remove any XML/URDF content
    lines = cleaned.split('\n')
    yaml_lines = []
    in_yaml = True
    
    for line in lines:
        if line.strip().startswith('<?xml') or line.strip().startswith('<robot'):
            in_yaml = False
            continue
        if line.strip().startswith('</robot>'):
            in_yaml = True
            continue
        if in_yaml and not line.strip().startswith('<'):
            yaml_lines.append(line)
    
    return '\n'.join(yaml_lines).strip()

def yaml_to_sdf(yaml_content: str) -> str:
    """Convert YAML model representation to SDF format"""
    
    model_data = yaml.safe_load(yaml_content)
    
    sdf_template = f"""<?xml version="1.0"?>
    <sdf version="1.7">
    <model name="{model_data['model']['name']}">
        <pose>0 0 0 0 0 0</pose>
        
        <!-- Links -->
    {_generate_links_sdf(model_data['model']['links'])}
        
        <!-- Joints -->
    {_generate_joints_sdf(model_data['model']['joints'])}
        
        <!-- Plugins -->
    {_generate_plugins_sdf(model_data['model'].get('plugins', []))}
    </model>
    </sdf>"""
    
    return sdf_template

def _generate_links_sdf(links: List[Dict]) -> str:
    """Generate SDF for links with proper inertial data handling"""
    links_sdf = []
    for link in links:
        # Handle missing inertial data with defaults
        if 'inertial' not in link:
            # Use reasonable defaults for small components
            mass = 0.1  # default mass for small parts
            inertia = [0.001, 0, 0, 0.001, 0, 0.001]  # default inertia
        else:
            mass = link['inertial']['mass']
            inertia = link['inertial']['inertia']
        
        link_sdf = f"""    <link name="{link['name']}">
        <pose>{' '.join(map(str, link['pose']))}</pose>
        <visual name="visual">
            <geometry>
            <mesh>
                <uri>{link['visual']['geometry']['uri']}</uri>
            </mesh>
            </geometry>
        </visual>
        <collision name="collision">
            <geometry>
            <mesh>
                <uri>{link['collision']['geometry']['uri']}</uri>
            </mesh>
            </geometry>
        </collision>
        <inertial>
            <mass>{mass}</mass>
            <inertia>
            <ixx>{inertia[0]}</ixx>
            <ixy>{inertia[1]}</ixy>
            <ixz>{inertia[2]}</ixz>
            <iyy>{inertia[3]}</iyy>
            <iyz>{inertia[4]}</iyz>
            <izz>{inertia[5]}</izz>
            </inertia>
        </inertial>
        </link>"""
        links_sdf.append(link_sdf)
    
    return '\n'.join(links_sdf)

def _generate_joints_sdf(joints: List[Dict]) -> str:
    """Generate SDF for joints"""
    joints_sdf = []
    for joint in joints:
        joint_sdf = f"""    <joint name="{joint['name']}" type="{joint['type']}">
    <parent>{joint['parent']}</parent>
    <child>{joint['child']}</child>
    <pose>{' '.join(map(str, joint['pose']))}</pose>
    <axis>
        <xyz>{' '.join(map(str, joint['axis']))}</xyz>
    </axis>
    </joint>"""
        joints_sdf.append(joint_sdf)
    
    return '\n'.join(joints_sdf)

def _generate_plugins_sdf(plugins: List[Dict]) -> str:
    """Generate SDF for plugins"""
    plugins_sdf = []
    for plugin in plugins:
        plugin_sdf = f"""    <plugin name="{plugin['name']}" filename="{plugin['filename']}">"""
        for key, value in plugin.get('parameters', {}).items():
            plugin_sdf += f"\n      <{key}>{value}</{key}>"
        plugin_sdf += "\n    </plugin>"
        plugins_sdf.append(plugin_sdf)
    
    return '\n'.join(plugins_sdf) if plugins_sdf else "    <!-- No plugins -->"


async def _list_required_meshes(assembly_plan: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extracts unique mesh file IDs and formats required by the assembly plan,
        and validates/resolves them to valid UUIDs if needed."""
    
    required_meshes = {}
    async with engine.connect() as conn:
        for part in assembly_plan.get('parts', []):
            mesh_id_or_name = part.get('mesh_file')
            mesh_format = part.get('mesh_format')
            part_name = part.get('name')
            
            if mesh_id_or_name and mesh_format:
                resolved_uuid = None
                # Check if the value is already a UUID (best case)
                try:
                    uuid.UUID(mesh_id_or_name)
                    resolved_uuid = mesh_id_or_name
                except ValueError:                    
                    path_like_name = f"%{mesh_id_or_name}%"
                    
                    # Query asset_files table to find a matching UUID based on path
                    result = await conn.execute(
                        text("SELECT id FROM asset_files WHERE path LIKE :path_name LIMIT 1"),
                        {"path_name": path_like_name}
                    )
                    row = await result.scalar_one_or_none()

                    if row:
                        resolved_uuid = str(row)
                    else:
                        prompt = '***' + part.get('name') + ' used in ' + assembly_plan.get('model_name', '')
                        continue  # Skip this mesh link insertion

                if resolved_uuid:
                    required_meshes[resolved_uuid] = {
                        "file_id": resolved_uuid,
                        "format": mesh_format,
                        "name": part_name
                    }
                    
        return list(required_meshes.values())


async def _hybrid_search_parts(llm_client, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Hybrid search combining vector similarity and semantic matching"""
    
    model_type = requirements.get('model_type', 'custom')
    key_components = requirements.get('key_components', [])
    primary_function = requirements.get('primary_function', '')
    
    # Generate embedding for semantic search
    search_query = f"{model_type} {primary_function} {requirements.get('mobility_type', '')}"
    query_embedding = await llm_client.get_embedding(search_query)  # Note: added await
    
    # Build hybrid search query
    query = """
        WITH semantic_matches AS (
        SELECT p.*, d.name as device_name, d.specs as device_specs,
            m.file_id as mesh_file, m.format as mesh_format,
            1 - (ev.embedding_vector <=> cast(:embedding as vector)) as similarity_score
        FROM parts p
        LEFT JOIN devices d ON p.device_id = d.id
        LEFT JOIN meshes m ON p.id = m.part_id
        LEFT JOIN embedding_vectors ev ON p.id = ev.resource_id AND ev.resource_type = 'part'
        WHERE ev.embedding_vector IS NOT NULL
        ORDER BY similarity_score DESC
        LIMIT 25
    ),
    category_matches AS (
        SELECT p.*, d.name as device_name, d.specs as device_specs,
                m.file_id as mesh_file, m.format as mesh_format,
                CASE 
                    WHEN p.category = ANY(:categories) THEN 1.0
                    WHEN :model_type = ANY(p.model_categories) THEN 0.8
                    ELSE 0.3
                END as category_score
        FROM parts p
        LEFT JOIN devices d ON p.device_id = d.id
        LEFT JOIN meshes m ON p.id = m.part_id
        WHERE p.category = ANY(:categories) OR :model_type = ANY(p.model_categories)
        LIMIT 25
    )
    SELECT DISTINCT ON (p.id) 
        p.id, p.name, p.description, p.category, p.model_categories,
        d.name as device_name, d.specs as device_specs,
        m.file_id as mesh_file, m.format as mesh_format,
        COALESCE(sm.similarity_score, 0) * 0.7 + COALESCE(cm.category_score, 0) * 0.3 as relevance_score
    FROM parts p
    LEFT JOIN devices d ON p.device_id = d.id
    LEFT JOIN meshes m ON p.id = m.part_id
    LEFT JOIN semantic_matches sm ON p.id = sm.id
    LEFT JOIN category_matches cm ON p.id = cm.id
    WHERE sm.id IS NOT NULL OR cm.id IS NOT NULL
    ORDER BY p.id, relevance_score DESC
    LIMIT 30
    """
    
    async with engine.connect() as conn:
        result = await conn.execute(
            text(query),
            {
                "embedding": query_embedding,
                "model_type": model_type,
                "categories": key_components if key_components else _get_default_categories(model_type)
            }
        )
        rows = await result.fetchall()
        return [dict(row._mapping) for row in rows]

def _get_default_categories(self, model_type: str) -> List[str]:
    """Get default part categories for model type"""
    categories = {
        'robot': ['chassis', 'wheel', 'motor', 'sensor', 'battery', 'manipulator'],
        'automobile': ['chassis', 'tire', 'suspension', 'engine', 'steering'],
        'drone': ['frame', 'motor', 'propeller', 'flight_controller', 'battery'],
        'human': ['torso', 'limb', 'joint', 'biomechanical'],
        'aircraft': ['fuselage', 'wing', 'engine', 'control_surface'],
        'flying_wing': ['wing_body', 'motor', 'elevon', 'flight_controller']
    }
    return categories.get(model_type, ['chassis', 'motor', 'sensor'])


async def _generate_model_files(llm_client, assembly_plan: Dict[str, Any]) -> Dict[str, str]:
    """Generate model files using self-hosted LLM and open-source tools"""
    model_files = {}
    model_name = assembly_plan['model_name']
    
    # Generate YAML content
    yaml_content = await _generate_yaml_content(llm_client, assembly_plan)  # Note: you need to pass llm_client here
    sdf_content = yaml_to_sdf(yaml_content)
    
    model_files['model.yaml'] = yaml_content
    model_files["model.sdf"] = sdf_content
    
    return model_files

