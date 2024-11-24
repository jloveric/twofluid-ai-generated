--Shock initial conditions
setup={
	--initialConditions = "brio_and_wu",
	initialConditions = "lua",
	algorithmStyle = "general",
	systemOfEquations = "five-moment-fluid-phm",
	dimensions = 2,
}

--You can use these variables here.
COPY = -1
PERIODIC = -2
FIXED = -3
CONTINUOUS = -4
CONDUCTING_WALL = -5
AXISYMMETRIC = -6
NO_INFLOW = -7
COIL = -8

universe = {
	grid = {iCells = 100, jCells = 1, kCells = 1, nGhost = 1},

	block = {iBlocks= 1, jBlocks = 1, kBlocks = 1},

	range = {xMin=-0.5, xMax=0.5, yMin=0, yMax=0.5, zMin=-1, zMax=1},

	bcs = {
		imbc = COPY,
		ipbc = COPY,
		jmbc = PERIODIC,
		jpbc = PERIODIC,
		kmbc = PERIODIC,
		kpbc = PERIODIC,
	},
}

constants = {
	ionCharge = 1.0,
	electronCharge = -1.0,
	ionMass = 1,
	electronMass = 5.447e-4,
	speedOfLight = 1.0,
	ionGamma = 1.66666,
	electronGamma = 1.66666,
	gamma = 1.66666,
	mass = 1.0,
	mu0	= 1,
	epsilon0 = 1,
	correctionSpeed = 0.0,
	electricCorrectionSpeed = 0.0,
	magneticCorrectionSpeed = 0.0,
	scalarResistivity = 0.0,
	relaxation = 0.0,
	resistivityModel = "constant",
	basementDensity = 0.0,
	basementNumberDensity = 0.0,
	basementPressure = 0.0,
}

initialConditions = {

	--Electron variables
	rhoe = function (x,y,z)
		if(x<0.0) then
			return 1.0*constants.electronMass;
		else
			return 0.125*constants.electronMass;
		end
	end,

	--Momentum
	mxe = function (x,y,z)
		return 0.0
	end,

	mye = function (x,y,z)
		return 0.0
	end,

	mze = function (x,y,z)
		return 0.0
	end,

	--Energy
	ene = function (x,y,z)
		if (x<0.0) then
			return 0.5e-4/(constants.gamma-1.0)
		else
			return 0.05e-4/(constants.gamma-1.0)
		end
	end,

	--Electron variables
	rhoi = function (x,y,z)
		if(x<0.0) then
			return 1.0*constants.ionMass;
		else
			return 0.125*constants.ionMass;
		end
	end,

	--Momentum
	mxi = function (x,y,z)
		return 0.0
	end,

	myi = function (x,y,z)
		return 0.0
	end,

	mzi = function (x,y,z)
		return 0.0
	end,

	--Energy
	eni = function (x,y,z)
		if (x<0.0) then
			return 0.5e-4/(constants.gamma-1.0)
		else
			return 0.05e-4/(constants.gamma-1.0)
		end
	end,

	--Fields
	bx = function(x,y,z)
		return 0.0075
	end,

	by = function(x,y,z)
		if(x<0.0) then
			return 0.01
		else
			return -0.01
		end
	end,

	bz = function(x,y,z)
		return 0.0
	end,

	ex = function(x,y,z)
		return 0.0
	end,

	ey = function(x,y,z)
		return 0.0
	end,

	ez = function(x,y,z)
		return 0.0
	end,

	bp = function(x,y,z)
		return 0.0
	end,

	ep = function(x,y,z)
		return 0.0
	end,
}

sources = {
	source={},
	region={},

	build = function ()
		--define local variables to reduce
		--complexity, these are actually references
		local source=sources.source
		local region=sources.region

		--allocate the sources.
		for i=1,2 do
			source[i]={};
			region[i]={}
		end

		source[1].kind = "constant"
		source[1].size = 8
		source[1].constant = 1.0
		source[1].component = "ez"
		
		region[1].kind = "block"
		region[1].size = 8
		region[1].range = {0, 1.0, 0.0, 1.0}
		region[1].component = "ez"

		--second source and region
		source[2].kind = "constant"
		source[2].size = 8
		source[2].constant = 1.0
		source[2].component = "ez"

		region[2].kind = "block"
		region[2].size = 8
		region[2].range = {0, 1.0, 0.0, 1.0}
		region[2].component = "ez"

		return;
	end,

} sources.build()

--print(sources.source[2].kind)

alpha = 0.0001
beta = 100

algorithm = {
	fluxType = "lax",
	orderSpace = 2,
	orderTime = 3,
	limiterStyle="characteristic",
	limiterFactor =
		{maxwell = 0.0, ions = 0.0, electrons = 0.0, mhd = 0.0, fluid=0.0},
}

timing={
	timeStep = 0.01,
	finalTime = 10.01,
	finalStep = 40000,
	cfl = 0.1,
}

restart = {
	restart = false,
	restartDumpCycle = 100,
}

dumping = {
	dumpTime = 0.1,
	output = matlab,
	dumpCycle = 100,
	dumpInterpolated = true,
	dumpPatchFormat = true,
	dumpCgns = true,
	diagnose = true,
}