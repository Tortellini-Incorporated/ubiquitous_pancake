use winit::{
	event::*,
	event_loop::{ControlFlow, EventLoop},
	window::{Window, WindowBuilder},
};
use winit::event::WindowEvent::{MouseInput, ModifiersChanged};
use rand::Rng;
use wgpu::util::DeviceExt;
use wgpu::TextureDataLayout;

struct State {
	surface: wgpu::Surface,
	device: wgpu::Device,
	queue: wgpu::Queue,
	sc_desc: wgpu::SwapChainDescriptor,
	swap_chain: wgpu::SwapChain,
	size: winit::dpi::PhysicalSize<u32>,
	render_pipeline: wgpu::RenderPipeline,

	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	num_indices: u32,

	diffuse_bind_group: wgpu::BindGroup,

	color: wgpu::Color,
	lmb_pressed: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
	position: [f32; 3],
	color: [f32; 3],
}

impl Vertex {
	fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
		wgpu::VertexBufferLayout {
			array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
			step_mode: wgpu::InputStepMode::Vertex,
			attributes: &[
				wgpu::VertexAttribute {
					format: wgpu::VertexFormat::Float3,
					offset: 0,
					shader_location: 0
				},
				wgpu::VertexAttribute {
					format: wgpu::VertexFormat::Float3,
					offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
					shader_location: 1
				}
			]
		}
	}
}

const VERTICES: &[Vertex] = &[
	Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
	Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
	Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];

const INDICES: &[u16] = &[
	0, 1, 2
];

impl State {
	async fn new(window: &Window) -> Self {
		let size = window.inner_size();

		let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
		let surface = unsafe { instance.create_surface(window) };
		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::HighPerformance,
				compatible_surface: Some(&surface),
			})
			.await
			.unwrap();
		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					label: None,
					features: wgpu::Features::empty(),
					limits: wgpu::Limits::default(),
				},
				None,
			)
			.await
			.unwrap();
		let sc_desc = wgpu::SwapChainDescriptor {
			usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
			format: wgpu::TextureFormat::Bgra8UnormSrgb,
			width: size.width,
			height: size.height,
			present_mode: wgpu::PresentMode::Fifo,
		};
		let swap_chain = device.create_swap_chain(&surface, &sc_desc);

		let vs_module = device.create_shader_module(&wgpu::include_spirv!("shader.vert.spv"));
		let fs_module = device.create_shader_module(&wgpu::include_spirv!("shader.frag.spv"));

		let vertex_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Vertex Buffer"),
				contents: bytemuck::cast_slice(VERTICES),
				usage: wgpu::BufferUsage::VERTEX,
			}
		);
		let index_buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Index Buffer"),
				contents: bytemuck::cast_slice(INDICES),
				usage: wgpu::BufferUsage::INDEX,
			}
		);
		let num_indices = INDICES.len() as u32;

		let diffuse_bytes = include_bytes!("../happy-tree.png");
		let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
		let diffuse_rgba = diffuse_image.as_rgba8().unwrap();

		use image::GenericImageView;
		let dimensions = diffuse_image.dimensions();

		let texture_size = wgpu::Extent3d {
			width: dimensions.0,
			height: dimensions.1,
			depth: 1
		};
		let diffuse_texture = device.create_texture(
			&wgpu::TextureDescriptor {
				label: Some("diffuse_texture"),
				size: texture_size,
				mip_level_count: 1,
				sample_count: 1,
				dimension: wgpu::TextureDimension::D2,
				format: wgpu::TextureFormat::Rgba8UnormSrgb,
				usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
			}
		);

		queue.write_texture(
			wgpu::TextureCopyView {
				texture: &diffuse_texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
			},
			diffuse_rgba,
			wgpu::TextureDataLayout {
				offset: 0,
				bytes_per_row:4 * dimensions.0,
				rows_per_image: dimensions.1,
			},
			texture_size,
		);/**//*
		let buffer = device.create_buffer_init(
			&wgpu::util::BufferInitDescriptor {
				label: Some("Temp Buffer"),
				contents: &diffuse_rgba,
				usage: wgpu::BufferUsage::COPY_SRC,
			}
		);

		let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
			label: Some("texture_buffer_copy_encoder"),
		});

		encoder.copy_buffer_to_texture(
			wgpu::BufferCopyView {
				buffer: &buffer,
				layout: TextureDataLayout {
					offset: 0,
					bytes_per_row: 3 * dimensions.0,
					rows_per_image: dimensions.1,
				}
			},
			wgpu::TextureCopyView {
				texture: &diffuse_texture,
				mip_level: 0,
				//array_layer: 0,
				origin: wgpu::Origin3d::ZERO,
			},
			texture_size,
		);

		queue.submit(std::iter::once(encoder.finish()));
		//end thing!*/

		let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
		let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
			address_mode_u: wgpu::AddressMode::ClampToEdge,
			address_mode_v: wgpu::AddressMode::ClampToEdge,
			address_mode_w: wgpu::AddressMode::ClampToEdge,
			mag_filter: wgpu::FilterMode::Linear,
			min_filter: wgpu::FilterMode::Nearest,
			mipmap_filter: wgpu::FilterMode::Nearest,
			..Default::default()
		});

		let texture_bind_group_layout = device.create_bind_group_layout(
			&wgpu::BindGroupLayoutDescriptor {
				entries: &[
					wgpu::BindGroupLayoutEntry {
						binding: 0,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::Texture {
							multisampled: false,
							view_dimension: wgpu::TextureViewDimension::D2,
							sample_type: wgpu::TextureSampleType::Uint,
						},
						count: None,
					},
					wgpu::BindGroupLayoutEntry {
						binding: 1,
						visibility: wgpu::ShaderStage::FRAGMENT,
						ty: wgpu::BindingType::Sampler {
							filtering: false,
							comparison: false,
						},
						count: None,
					},
				],
				label: Some("texture_bind_group_layout"),
			}
		);

		let diffuse_bind_group = device.create_bind_group(
			&wgpu::BindGroupDescriptor {
				layout: &texture_bind_group_layout,
				entries: &[
					wgpu::BindGroupEntry {
						binding: 0,
						resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
					},
					wgpu::BindGroupEntry {
						binding: 1,
						resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
					}
				],
				label: Some("diffuse_binding_group"),
			}
		);

		let render_pipeline_layout = device.create_pipeline_layout(
			&wgpu::PipelineLayoutDescriptor {
				label: Some("Render Pipeline Layout"),
				bind_group_layouts: &[&texture_bind_group_layout],
				push_constant_ranges: &[],
			}
		);
		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Render Pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &vs_module,
				entry_point: "main",
				buffers: &[
					Vertex::desc(),
				]
			},
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: wgpu::CullMode::Back,
				polygon_mode: wgpu::PolygonMode::Fill
			},
			depth_stencil: None,
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: !0,
				alpha_to_coverage_enabled: false
			},
			fragment: Some(wgpu::FragmentState {
				module: &fs_module,
				entry_point: "main",
				targets: &[
					wgpu::ColorTargetState {
						format: sc_desc.format,
						color_blend: wgpu::BlendState::REPLACE,
						alpha_blend: wgpu::BlendState::REPLACE,
						write_mask: wgpu::ColorWrite::ALL,
					}
				],
			}),
		});

		Self {
			surface,
			device,
			queue,
			sc_desc,
			swap_chain,
			size,
			render_pipeline,
			vertex_buffer,
			index_buffer,
			num_indices,
			diffuse_bind_group,

			color: wgpu::Color {
				r: 0.1,
				g: 0.2,
				b: 0.3,
				a: 1.0,
			},
			lmb_pressed: false
		}
	}

	fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
		self.size = new_size;
		self.sc_desc.width = new_size.width;
		self.sc_desc.height = new_size.height;
		self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
	}

	fn input(&mut self, event: &WindowEvent) -> bool {
		if let WindowEvent::MouseInput { device_id, state, button, modifiers } = event {
			if *button == MouseButton::Left {
				if *state == ElementState::Pressed {
					self.lmb_pressed = true;
				} else if *state == ElementState::Released {
					if self.lmb_pressed {
						self.color = wgpu::Color {
							r: rand::random(),
							g: rand::random(),
							b: rand::random(),
							a: 1.0
						};
						self.lmb_pressed = false;
					}
				}
			}
		}
		false
	}

	fn update(&mut self) {}

	fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
		let frame = self.swap_chain.get_current_frame()?.output;
		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("Render Encoder"),
			});

		let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: None,
			color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
				attachment: &frame.view,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(self.color),
					store: true,
				},
			}],
			depth_stencil_attachment: None,
		});
		render_pass.set_pipeline(&self.render_pipeline);
		render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
		render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
		render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
		render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

		drop(render_pass);

		self.queue.submit(std::iter::once(encoder.finish()));

		Ok(())
	}
}

fn main() {
	use futures::executor::block_on;

	env_logger::init();
	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	let mut state = block_on(State::new(&window));

	event_loop.run(move |event, _, control_flow| match event {
		Event::WindowEvent {
			ref event,
			window_id,
		} if window_id == window.id() => {
			if !state.input(event) {
				match event {
					WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
					WindowEvent::KeyboardInput { input, .. } => match input {
						KeyboardInput {
							state: ElementState::Pressed,
							virtual_keycode: Some(VirtualKeyCode::Escape),
							..
						} => *control_flow = ControlFlow::Exit,
						_ => {}
					},
					WindowEvent::Resized(physical_size) => {
						state.resize(*physical_size);
					}
					WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
						state.resize(**new_inner_size);
					}
					_ => {}
				}
			}
		}
		Event::RedrawRequested(_) => {
			state.update();
			match state.render() {
				Ok(_) => {}
				Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
				Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
				Err(e) => eprintln!("{:?}", e),
			}
		}
		Event::MainEventsCleared => {
			window.request_redraw();
		}
		_ => {}
	});
}
