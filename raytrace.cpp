//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"
#include <vector>

const float epsilon = 0.0001f;

struct Hit {
    float t;
    vec3 position, normal;
    bool cone = false;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

struct Plane : public Intersectable{
    vec3 point;
    vec3 normal;

    Plane(vec3 _point, vec3 _normal) {
        point = _point;
        normal = _normal;
    }

    Plane(vec3 vec) {
        point = vec;
        normal = normalize(vec);
    }

    void translate(vec3 d) {
        point = point + d;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        float alignment = dot(ray.dir, normal);
        vec3 to_point = point - ray.start;
        float t = dot(to_point, normal) / alignment;
        hit.t = t;
        hit.position = ray.start + t * ray.dir;
        hit.normal = normal;
        return hit;
    }
};

struct Solid : public Intersectable {
    std::vector<Plane> faces;
    bool inverted;

    Solid(const std::vector<Plane>& _faces, bool _inverted = false)
        : faces(_faces), inverted(_inverted) {}

    void translate(vec3 d) {
        for (Plane& p : faces) {
            p.translate(d);
        }
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        for (Plane& face : faces) {
            hit = face.intersect(ray);
            float alignment = dot(ray.dir, hit.normal);
            bool facing = (alignment > 0) == inverted;
            bool correct = true;
            if (hit.t > epsilon && facing) {
                for (Plane& f : faces) {
                    vec3 from_plane = hit.position - f.point;
                    if (&f != &face && dot(from_plane, f.normal) > epsilon) {
                        correct = false;
                        break;
                    }
                }
                if (correct) {
                    return hit;
                }
            }
        }
        return Hit();
    }
};

struct Cone : public Intersectable {
    vec3 tip;
    vec3 axis;
    float cos2;
    float height;
    vec3 light_position;
    vec3 light_color;

    Cone(vec3 _tip, vec3 _axis, float _cos2, float _height, vec3 color)
        : tip(_tip), axis(_axis), cos2(_cos2), height(_height), light_color(color) {
        light_position = tip + 5 * epsilon * axis;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        hit.cone = true;
        vec3 r = ray.start - tip;
        float ad = dot(axis, ray.dir);
        float ar = dot(axis, r);
        float a = ad * ad - cos2 * dot(ray.dir, ray.dir);
        float m = (cos2 * dot(r, ray.dir) - ad * ar) / a;
        float p = (ar * ar - cos2 * dot(r, r)) / a;
        float disc = m * m - p;
        if (disc > 0) {
            float diff = sqrtf(disc);
            float x = m - diff;
            if (x < 0) {
                x = m + diff;
            }

            vec3 pos = ray.start + x * ray.dir;
            float h = dot(pos - tip, axis);
            if (h > 0 && h < height) {
                hit.t = x;
                hit.position = pos;
                vec3 along = pos - tip;

                vec3 normal = 2 * dot(along, axis) * axis - 2 * cos2 * along;
                hit.normal = normalize(normal);
            }
            else {
                x = m + diff;
                vec3 pos = ray.start + x * ray.dir;
                float h = dot(pos - tip, axis);
                if (h > 0 && h < height) {
                    hit.t = x;
                    hit.position = pos;
                    vec3 along = pos - tip;

                    vec3 normal = 2 * dot(along, axis) * axis - 2 * cos2 * along;
                    hit.normal = normalize(normal);
                }
            }
        }
        return hit;
    }
};

class Camera {
    vec3 eye, lookat, right, up;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
};

float rnd() { return (float)rand() / static_cast<float>(RAND_MAX); }

std::vector<Cone*> lights = std::vector<Cone*>();

class Scene {
    std::vector<Intersectable *> objects;
    vec3 La;
public:
    Camera camera;
    void build() {
        vec3 eye = vec3(sinf(M_PI / 4.0f), 0, cosf(M_PI / 4.0f))*2;
        vec3 vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.0f, 0.0f, 0.0f);

        std::vector<Plane>faces1 = std::vector<Plane>();
        faces1.push_back(Plane(vec3(0,-1,0)*0.5, vec3(0,-1,0)));
        faces1.push_back(Plane(vec3(0,1,0)*0.5,vec3(0,1,0)));
        faces1.push_back(Plane(vec3(-1,0,0)*0.5,vec3(-1,0,0)));
        faces1.push_back(Plane(vec3(1,0,0)*0.5,vec3(1,0,0)));
        faces1.push_back(Plane(vec3(0,0,1)*0.5,vec3(0,0,1)));
        faces1.push_back(Plane(vec3(0,0,-1)*0.5,vec3(0,0,-1)));
        Solid* cube = new Solid(faces1, true);
        // cube->translate(vec3(0.5,-0.5,0));
        objects.push_back(cube);

        std::vector<Plane> faces2 = std::vector<Plane>();
        float rphi = (sqrtf(5) - 1) / 2.0f;
        float size = 0.1;
        faces2.push_back(Plane(size * vec3(rphi, 0, 1)));
        faces2.push_back(Plane(size * vec3(-rphi, 0, 1)));
        faces2.push_back(Plane(size * vec3(rphi, 0, -1)));
        faces2.push_back(Plane(size * vec3(-rphi, 0, -1)));
        faces2.push_back(Plane(size * vec3(1, rphi, 0)));
        faces2.push_back(Plane(size * vec3(1, -rphi, 0)));
        faces2.push_back(Plane(size * vec3(-1, rphi, 0)));
        faces2.push_back(Plane(size * vec3(-1, -rphi, 0)));
        faces2.push_back(Plane(size * vec3(0, 1, rphi)));
        faces2.push_back(Plane(size * vec3(0, 1, -rphi)));
        faces2.push_back(Plane(size * vec3(0, -1, rphi)));
        faces2.push_back(Plane(size * vec3(0, -1, -rphi)));
        Solid* dodeca = new Solid(faces2);
        dodeca->translate(vec3(0.2, size - 0.5, -0.3));
        objects.push_back(dodeca);
        
        std::vector<Plane> faces3 = std::vector<Plane>();
        float phi = (sqrtf(5) + 1) / 2.0f;
        size = 0.1;
        faces3.push_back(Plane(size * vec3(1, 1, 1)));
        faces3.push_back(Plane(size * vec3(1, 1, -1)));
        faces3.push_back(Plane(size * vec3(1, -1, 1)));
        faces3.push_back(Plane(size * vec3(1, -1, -1)));
        faces3.push_back(Plane(size * vec3(-1, 1, 1)));
        faces3.push_back(Plane(size * vec3(-1, 1, -1)));
        faces3.push_back(Plane(size * vec3(-1, -1, 1)));
        faces3.push_back(Plane(size * vec3(-1, -1, -1)));
        faces3.push_back(Plane(size * vec3(0, phi, 1 / phi)));
        faces3.push_back(Plane(size * vec3(0, phi, -1 / phi)));
        faces3.push_back(Plane(size * vec3(0, -phi, 1 / phi)));
        faces3.push_back(Plane(size * vec3(0, -phi, -1 / phi)));
        faces3.push_back(Plane(size * vec3(phi, 1 / phi, 0)));
        faces3.push_back(Plane(size * vec3(phi, -1 / phi, 0)));
        faces3.push_back(Plane(size * vec3(-phi, 1 / phi, 0)));
        faces3.push_back(Plane(size * vec3(-phi, -1 / phi, 0)));
        faces3.push_back(Plane(size * vec3(1 / phi, 0, phi)));
        faces3.push_back(Plane(size * vec3(-1 / phi, 0, phi)));
        faces3.push_back(Plane(size * vec3(1 / phi, 0, -phi)));
        faces3.push_back(Plane(size * vec3(-1 / phi, 0, -phi)));
        Solid* icosa = new Solid(faces3);
        icosa->translate(vec3(-0.1, size * phi - 0.5, 0.1));
        objects.push_back(icosa);

        Cone * c1 = new Cone(vec3(0,1,0)*0.5, vec3(0,1,0)*-1, 0.85, 0.15f, vec3(0.3,0,0));
        Cone * c2 = new Cone(vec3(-1,0,0)*0.5, vec3(-1,0,0)*-1, 0.75, 0.15f, vec3(0,0.3,0));
        Cone * c3 = new Cone(vec3(0,0,-1)*0.5, vec3(0,0,-1)*-1, 0.95, 0.15f, vec3(0,0,0.3));
        objects.push_back(c1);
        lights.push_back(c1);
        objects.push_back(c2);
        lights.push_back(c2);
        objects.push_back(c3);
        lights.push_back(c3);
        // objects.push_back(new Plane(vec3(0,-1,0)*0.5, vec3(0,-1,0)));
        // objects.push_back(new Plane(vec3(0,-1,0)*0.5, vec3(0,1,0)));
    }

    void render(std::vector<vec4>& image) {
        for (uint Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (uint X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    vec3 trace(Ray ray) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;
        vec3 outRadiance = La;
        outRadiance = vec3(0.2f, 0.2f, 0.2f) * (1 - dot(hit.normal, ray.dir));
        for (Cone* c : lights) {
            vec3 to_object = hit.position + epsilon * hit.normal - c->light_position;
            float distance = length(to_object);
            Ray ray = Ray(c->light_position, normalize(to_object));
            Hit first = firstIntersect(ray);
            if (first.t > distance) {
                outRadiance = outRadiance + c->light_color / (distance * distance);
            }
        }
        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
    #version 330
    precision highp float;

    layout(location = 0) in vec2 cVertexPosition;   // Attrib Array 0
    out vec2 texcoord;

    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;                            // -1,1 to 0,1
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);         // transform to clipping space
    }
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
    #version 330
    precision highp float;

    uniform sampler2D textureUnit;
    in  vec2 texcoord;          // interpolated texture coordinates
    out vec4 fragmentColor;     // output that goes to the raster memory as told by glBindFragDataLocation

    void main() {
        fragmentColor = texture(textureUnit, texcoord); 
    }
)";

class FullScreenTexturedQuad {
    unsigned int vao;   // vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
        : texture(windowWidth, windowHeight, image) 
    {
        glGenVertexArrays(1, &vao); // create 1 vertex array object
        glBindVertexArray(vao);     // make it active

        unsigned int vbo;       // vertex buffer objects
        glGenBuffers(1, &vbo);  // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };   // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);     // copy to that part of the memory which is not modified 
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao); // make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %ld milliseconds\n", (timeEnd - timeStart));

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                  // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    Hit hit = scene.firstIntersect(scene.camera.getRay(pX, windowHeight-pY));
    if (hit.t > 0 && !hit.cone) {
        Cone * closest = lights[0];
        for (Cone* c : lights) {
            if (length(c->tip - hit.position) < length(closest->tip - hit.position)) {
                closest = c;
            }
        }
        closest->tip = hit.position;
        closest->axis = hit.normal;
        closest-> light_position = hit.position + 5 * epsilon * hit.normal;
        std::vector<vec4> image(windowWidth * windowHeight);
        scene.render(image);
        fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
        glutPostRedisplay();
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
