"""
A small hand-written lighting shader.

Panda3D's automatic shader generator (setShaderAuto) renders everything
unlit in the macOS core-profile OpenGL context, so this replaces it: soft
diffuse light from the scene's lights, a gentle rim highlight that gives the
rounded shapes a friendly "toy" look, and Panda's own light/ambient inputs so
the lights set up in scene.py drive it.
"""

from panda3d.core import Shader

VERTEX = """
#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
out vec3 v_normal;
out vec3 v_view_pos;
out vec4 v_color;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_view_pos = (p3d_ModelViewMatrix * p3d_Vertex).xyz;
    v_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    v_color = p3d_Color;
}
"""

FRAGMENT = """
#version 150
uniform vec4 p3d_ColorScale;
uniform struct p3d_LightModelParameters { vec4 ambient; } p3d_LightModel;
uniform struct p3d_LightSourceParameters {
    vec4 color;
    vec4 position;
} p3d_LightSource[2];
in vec3 v_normal;
in vec3 v_view_pos;
in vec4 v_color;
out vec4 p3d_FragColor;
void main() {
    vec4 base = v_color * p3d_ColorScale;
    vec3 n = normalize(v_normal);
    vec3 light = p3d_LightModel.ambient.rgb;
    for (int i = 0; i < 2; ++i) {
        vec4 lp = p3d_LightSource[i].position;
        vec3 to_light = lp.w == 0.0 ? normalize(lp.xyz) : normalize(lp.xyz - v_view_pos);
        // Wrapped diffuse: softer falloff than plain Lambert, no harsh black sides.
        float d = max((dot(n, to_light) + 0.35) / 1.35, 0.0);
        light += p3d_LightSource[i].color.rgb * d;
    }
    float rim = pow(1.0 - max(dot(n, normalize(-v_view_pos)), 0.0), 3.0) * 0.18;
    p3d_FragColor = vec4(base.rgb * light + rim, base.a);
}
"""


def apply_lighting(render):
    render.setShader(Shader.make(Shader.SL_GLSL, VERTEX, FRAGMENT))
