#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>


#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NormalizationLib.h"
#include "DispLib.h"
#include "histograms.h"
#include "gradient.h"
#include "RegionU16Lib.h"

#include "mazdaroi.h"
#include "mazdaroiio.h"

#include <tiffio.h>


using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;
//------------------------------------------------------------------------------------------------------------------------------
string MatPropetiesAsText(Mat Im)
{
    string Out ="Image properties: ";
    Out += "max x = " + to_string(Im.cols);
    Out += ", max y = " + to_string(Im.rows);
    Out += ", # channels = " + to_string(Im.channels());
    Out += ", depth = " + to_string(Im.depth());

    switch(Im.depth())
    {
    case CV_8U:
        Out += "CV_8U";
        break;
    case CV_8S:
        Out += "CV_8S";
        break;
    case CV_16U:
        Out += "CV_16U";
        break;
    case CV_16S:
        Out += "CV_16S";
        break;
    case CV_32S:
        Out += "CV_32S";
        break;
    case CV_32F:
        Out += "CV_32F";
        break;
    case CV_64F:
        Out += "CV_64F";
        break;
    default:
        Out += " unrecognized ";
    break;
    }
    return Out;
}
//------------------------------------------------------------------------------------------------------------------------------
string TiffFilePropetiesAsText(string FileName)
{
    float xRes,yRes;
    uint32 imWidth, imLength;
    uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);


        Out += "max x = " + to_string(imLength);
        Out += ", max y = " + to_string(imWidth);
        Out += ", ResUnit = " + to_string(resolutionUnit);
        Out += ", xRes = " + to_string(1.0/xRes);
        Out += ", yRes = " + to_string(1.0/yRes);
        TIFFClose(tifIm);
    }
    else
        Out += " improper file ";
    //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH, &width);

    return Out;
}
//------------------------------------------------------------------------------------------------------------------------------

bool GetTiffProperties(string FileName, float &xRes, float &yRes)
{
    //float xRes,yRes;
    //uint32 imWidth, imLength;
    //uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        //TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        //TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);

        TIFFClose(tifIm);
        return 1;
    }
    else
    {
        xRes = 1.0;
        yRes = 1.0;
        return 0;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
cv::Mat MaskBackround(cv::Mat ImIn)
{
    Mat Mask;
    Mask.release();
    if(ImIn.channels() != 3)
    {
        return Mask;
    }

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;
    int maxXY = maxX * maxY;
    Mask = Mat::zeros(maxY, maxX, CV_16U);

    uint16 *wMask = (uint16 *)Mask.data;

    unsigned char *wImIn = (unsigned char *)ImIn.data;

    for(int i = 0; i < maxXY; i++)
    {
        unsigned char B = *wImIn;
        wImIn++;
        unsigned char G = *wImIn;
        wImIn++;
        unsigned char R = *wImIn;
        wImIn++;

        if(R > 95 && G > 40 && B > 20 && R - G > 15 && R > G && R > B)
            *wMask = 1;
        wMask++;
    }

    return Mask;
}
//------------------------------------------------------------------------------------------------------------------------------
void ShowsScaledImage2(Mat Im, string ImWindowName, double displayScale, bool imRotate)
{
    if(Im.empty())
    {
        return;
    }

    Mat ImToShow;


    ImToShow = Im.clone();

    if (displayScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    if(imRotate)
        rotate(ImToShow,ImToShow, ROTATE_90_CLOCKWISE);
    imshow(ImWindowName, ImToShow);

}
//------------------------------------------------------------------------------------------------------------------------------
void GetLesionMask(Mat Im, Mat Mask, int par1, int par2)
{
    int maxXY = Im.cols * Im.rows;

    uint8_t *wIm;
    uint16_t *wMask;
    wMask = (uint16_t*)Mask.data;
    wIm = (uint8_t*)Im.data;
    for(int i = 0; i< maxXY; i++)
    {
        uint8_t b = *wIm;
        wIm++;
        uint8_t g = *wIm;
        wIm++;
        uint8_t r = *wIm;
        wIm++;

        if(*wMask == 0 || b > par1 || g > par2)
        {
            *wMask = 0;
        }
        wMask++;
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void GetLesionMaskFromHStip(Mat Im, Mat Mask)
{

    int stripExpansion = 1;

    int maxX = Im.cols;
    int maxY = Im.rows;

    int maxX3 = maxX*3 - 3;

    Mat ImStrip;
    Mat MaskStrip;
    int startX = stripExpansion;
    int stripSizeX = 1+ 2 * stripExpansion;
    int stopX = maxX - stripExpansion;
/*
    unsigned char *wIm = (unsigned char *)Im.data;
    for(int x = 0; x < maxX; x++)
    {
        uint16_t *wMask = (uint16_t *)Mask.data + x;
        wIm = (unsigned char *)Im.data + (x * 3);
        for(int y = 0; y < maxY; y++)
        {
            uint8_t b = *wIm;
            wIm++;
            uint8_t g = *wIm;
            wIm++;
            uint8_t r = *wIm;
            wIm++;


            *wMask = r;

            wMask += maxX;
            wIm += maxX3;
        }
    }
return;
*/
    for(int x = startX; x < stopX; x++)
    {

        Im(Rect(x - stripExpansion, 0, stripSizeX, maxY)).copyTo(ImStrip);

        Mask(Rect(x - stripExpansion, 0, stripSizeX, maxY)).copyTo(MaskStrip);


        HistogramRGB HistogramStrip;

        HistogramStrip.FromMat(ImStrip,MaskStrip, 1);


        uint16_t *wMask = (uint16_t *)Mask.data + x;
        unsigned char *wIm = (unsigned char *)Im.data + (x * 3);

        uint8 thresholdB = HistogramStrip.GetMeanB() - (HistogramStrip.GetMaxB() - HistogramStrip.GetMeanB());
        uint8 thresholdG = HistogramStrip.GetMeanG() - (HistogramStrip.GetMaxG() - HistogramStrip.GetMeanG());
        int count = HistogramStrip.GetCount();

        //continue;

        for(int y = 0; y < maxY; y++)
        {
            uint8_t b = *wIm;
            wIm++;
            uint8_t g = *wIm;
            wIm++;
            uint8_t r = *wIm;
            wIm++;
            if(*wMask == 0 || b > thresholdB || g > thresholdG || count < 200)
            {
                *wMask = 0;
            }

            wMask += maxX;
            wIm += maxX3;
        }

    }


}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          constructor Destructor
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->comboBoxOutputMode->addItem("Gray");
    ui->comboBoxOutputMode->addItem("B");
    ui->comboBoxOutputMode->addItem("G");
    ui->comboBoxOutputMode->addItem("R");
    ui->comboBoxOutputMode->setCurrentIndex(0);

    ui->comboBoxGradient->addItem("Morfological");
    ui->comboBoxGradient->addItem("Up");
    ui->comboBoxGradient->addItem("Down");
    ui->comboBoxGradient->setCurrentIndex(0);

}

MainWindow::~MainWindow()
{
    delete ui;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          CLASS FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::OpenImageFolder()
{
    if (!exists(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " not exists "));
        ImageFolder = "d:\\";
    }
    if (!is_directory(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " This is not a directory path "));
        ImageFolder = "C:\\Data\\";
    }
    ui->lineEditImageFolder->setText(QString::fromStdString(ImageFolder.string()));
    ui->listWidgetImageFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(ImageFolder))
    {
        regex FilePattern(ui->lineEditRegexImageFile->text().toStdString());
        if (!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern ))
            continue;
        path PathLocal = FileToProcess.path();
        if (!exists(PathLocal))
        {
            ui->textEditOut->append(QString::fromStdString(PathLocal.filename().string() + " File not exists" ));
            break;
        }
        ui->listWidgetImageFiles->addItem(PathLocal.filename().string().c_str());
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ReadImage()
{
    if(ui->checkBoxAutocleanOut->checkState())
        ui->textEditOut->clear();
    int flags;
    if(ui->checkBoxLoadAnydepth->checkState())
        flags = CV_LOAD_IMAGE_ANYDEPTH;
    else
        flags = IMREAD_COLOR;
    ImIn = imread(FileName, flags);
    if(ImIn.empty())
    {
        ui->textEditOut->append("improper file");
        return;
    }

    path FileNamePath(FileName);
    string extension = FileNamePath.extension().string();

    if((extension == ".tif" || extension == ".tiff") && ui->checkBoxShowTiffInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(TiffFilePropetiesAsText(FileName)));

    if(ui->checkBoxShowMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImIn)));

    ProcessImages();



}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;


    ImToShow = Im.clone();

    displayScale = pow(double(ui->spinBoxScaleBase->value()), double(ui->spinBoxScalePower->value()));
    if (displayScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    if(ui->checkBoxImRotate->checkState())
        rotate(ImToShow,ImToShow, ROTATE_90_CLOCKWISE);
    imshow(ImWindowName, ImToShow);

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowImages()
{
    if(ui->checkBoxShowInput->checkState())
        ShowsScaledImage(ImIn, "Input Image");

    if(ui->checkBoxShowGray->checkState())
        ShowsScaledImage(ShowImage16Gray(ImGray,ui->doubleSpinBoxFixMinDisp->value(),
                                         ui->doubleSpinBoxFixMaxDisp->value()), "Gray Image");

    if(ui->checkBoxShowPC->checkState())
        ShowsScaledImage(ShowImage16PseudoColor(ImGray,ui->doubleSpinBoxFixMinDisp->value(),
                                         ui->doubleSpinBoxFixMaxDisp->value()), "PC Image");

    if(ui->checkBoxShowGradient->checkState() && ui->checkBoxProcessGradient->checkState())
        ShowsScaledImage(ShowImage16PseudoColor(ImGradient,ui->doubleSpinBoxFixMinDispGrad->value(),
                                         ui->doubleSpinBoxFixMaxDispGrad->value()), "Gradient Image");

    if(ui->checkBoxShowMask->checkState())
        ShowsScaledImage(ShowRegion(Mask), "Mask");
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ProcessImages()
{
    if(ImIn.empty())
    {

        return;
    }
    Mat ImGrayTemp;
    if(ui->checkBoxLoadAnydepth->checkState())
    {
        ImGrayTemp = ImIn;
    }
    else
    {
        switch(ui->comboBoxOutputMode->currentIndex())
        {
            case 1:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[0];
            }
            break;

            case 2:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[1];
            }
            break;
            case 3:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[2];
            }
        break;
        default:
            cvtColor(ImIn, ImGrayTemp, CV_BGR2GRAY);
        break;
        }
    }

    ImGrayTemp.convertTo(ImGray,CV_16U);

    if(ui->checkBoxProcessGradient->checkState())
    {
        switch(ui->comboBoxGradient->currentIndex())
        {
        case 1:
            ImGradient = GradientUP(ImGray);
            ImGradient.convertTo(ImGradient,CV_16U,10);
        break;
        case 2:
            ImGradient = GradientDown(ImGray);
            ImGradient.convertTo(ImGradient,CV_16U,10);
        break;
        default:
            ImGradient = GradientMorph(ImGray,ui->spinBoxGradientSchape->value())*10;
        break;
        }
        MaskHair = Threshold16(ImGradient, ui->spinBoxGradThreshold->value());

    }

    if(ui->checkBoxMaskBackGround->checkState() && ImIn.channels() == 3)
    {
        Mask = MaskBackround(ImIn);
    }
    else
    {
        Mask = Mat::ones(ImIn.rows,ImIn.cols, CV_16U);
    }

    if(ui->checkBoxProcessGradient->checkState())
        MaskMaskInv(Mask, MaskHair);

    if(ui->checkBoxProcessTile->checkState())
    {
        int tileStepX;
        int tileStepY;
        int tileSizeY = ui->spinBoxTileSize->value();
        tileStepY =tileSizeY/2;
        int tileSizeX = ui->spinBoxTileSizeX->value();

        if (tileSizeX < 4)
            tileStepX =tileSizeX;
        else
            tileStepX =tileSizeX/2;

        TileImVector.clear();
        TileMaskVector.clear();
        TilePositionVector.clear();

        int maxX = ImIn.cols;
        int maxY = ImIn.rows;

        int limX = maxX - tileSizeX;
        int limY = maxY - tileSizeY;

        for(int y = 0; y < limY; y += tileStepY)
        {
            for(int x = 0; x < limX; x += tileStepX)
            {
                Mat TileMask;
                Mask(Rect(x, y, tileSizeX, tileSizeY)).copyTo(TileMask);

                uint16 *wTileMask = (uint16 *)TileMask.data;
                int tileMaxXY = tileSizeX * tileSizeY;
                int tileMaskCount = 0;
                for(int i = 0; i < tileMaxXY; i++)
                {
                    if(*wTileMask)
                        tileMaskCount++;
                    wTileMask++;
                }
                int tileMaskThreshold = tileMaxXY * 80/100;
                if(tileMaskCount > tileMaskThreshold)
                {
                    TileMaskVector.push_back(TileMask);
                    Mat TileIm;
                    ImIn(Rect(x, y, tileSizeX, tileSizeY)).copyTo(TileIm);
                    TileImVector.push_back(TileIm);
                    Point TilePosition = Point(x,y);
                    TilePositionVector.push_back(TilePosition);
                }
            }
        }
        ui->lineEditTileCount->setText(QString::fromStdString(to_string(TileImVector.size())));
        ui->spinBoxTileToProcess->setMaximum(TileImVector.size() - 1);

        ProcessTile();
    }


    ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ProcessTile()
{
    if(!ui->checkBoxProcessTile->checkState())
        return;


    if(TileImVector.empty())
        return;
    if(TileMaskVector.empty())
        return;
    if(TileImVector.size() != TileMaskVector.size())
        return;
    int tileNr = ui->spinBoxTileToProcess->value();
    if(tileNr >= TileMaskVector.size())
        return;

    Mat TileIm = TileImVector[tileNr];
    Mat TileMask = TileMaskVector[tileNr];
    Point TilePosition = TilePositionVector[tileNr];
    int tileSizeY = ui->spinBoxTileSize->value();
    int tileSizeX = ui->spinBoxTileSizeX->value();
    if(ui->checkBoxShowTile->checkState())
    {
        double tileScale = ui->doubleSpinBoxTileScale->value();
        ShowsScaledImage2(TileIm,"Tile Im",tileScale,false);
        ShowsScaledImage2(ShowRegion(TileMask),"Tile mask",tileScale,false);
    }
    if(ui->checkBoxShowTileOnImage->checkState())
    {

        Mat ImToShow;
        ImIn.copyTo(ImToShow);
        rectangle(ImToShow, Rect(TilePosition.x,TilePosition.y, tileSizeX, tileSizeY), Scalar(0.0, 255.0, 0.0, 0.0), 4);
        ShowsScaledImage(ImToShow, "Tile On Image");

    }
    Mat LesionMask;
    HistogramRGB HistogramTile;
    TileMask.copyTo(LesionMask);
    if(ui->checkBoxShowHist->checkState() || !ui->checkBoxHorisontalLineMode->checkState())
    {
        HistogramTile.FromMat(TileIm,TileMask, 1);
    }

    if(ui->checkBoxHorisontalLineMode->checkState())
    {
        GetLesionMaskFromHStip(TileIm,LesionMask);
        imshow("XXXXXXXXXXXXX", ShowImage16Gray(LesionMask,0,255));

    }
    else
    {
        uint8 thresholdB = HistogramTile.GetMeanB() - (HistogramTile.GetMaxB() - HistogramTile.GetMeanB());
        uint8 thresholdG = HistogramTile.GetMeanG() - (HistogramTile.GetMaxG() - HistogramTile.GetMeanG());
        string outText = "thB = " + to_string(thresholdB) + " thG = " + to_string(thresholdG);
        outText += " MeanB = " + to_string(HistogramTile.GetMeanB());
        outText += " MaxB = " + to_string(HistogramTile.GetMaxB());
        ui->textEditOut->append(QString::fromStdString(outText));
        GetLesionMask(TileIm,LesionMask, thresholdB, thresholdG);
    }

    if(ui->checkBoxShowHist->checkState())
    {

        imshow("Histogram from Tile" ,HistogramTile.PlotRGB(ui->spinBoxHistScaleHeight->value(),
                                     ui->spinBoxHistScaleCoef->value(),
                                     ui->spinBoxHistBarWidth->value()));



        //uint8 thresholdB = HistogramTile.meanB - (HistogramTile.maxB - HistogramTile.meanB);


    }


    if(ui->checkBoxShowLesionMask->checkState())
    {
        double tileScale = ui->doubleSpinBoxTileScale->value();
        ShowsScaledImage2(ShowRegion(LesionMask),"Lesion mask",tileScale,false);
    }

}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          Slots
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

void MainWindow::on_pushButtonOpenImageFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(QString::fromStdString(ImageFolder.string()));

    if(dialog.exec())
    {
        ImageFolder = dialog.directory().path().toStdWString();
    }
    else
        return;
    OpenImageFolder();
}

void MainWindow::on_lineEditRegexImageFile_returnPressed()
{
    OpenImageFolder();
}

void MainWindow::on_listWidgetImageFiles_currentTextChanged(const QString &currentText)
{

    path fileToOpen = ImageFolder;
    fileToOpen.append(currentText.toStdString());
    FileName = fileToOpen.string();
    ReadImage();
    path fileToOpenTxt = ImageFolder;
    fileToOpenTxt.append(fileToOpen.stem().string() +".txt");
    FileNameTxt = fileToOpenTxt.string();
    ui->textEditOut->append(QString::fromStdString(FileNameTxt));
}

void MainWindow::on_checkBoxShowInput_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_checkBoxShowTiffInfo_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxShowMatInfo_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxAutocleanOut_stateChanged(int arg1)
{
    ReadImage();
}

void MainWindow::on_checkBoxLoadAnydepth_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxShowOutput_toggled(bool checked)
{
    ShowImages();
}


void MainWindow::on_spinBoxScalePower_valueChanged(int arg1)
{
    ShowImages();
    ProcessTile();
}

void MainWindow::on_spinBoxScaleBase_valueChanged(int arg1)
{
    ShowImages();
    ProcessTile();
}

void MainWindow::on_checkBoxImRotate_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_pushButtonOpenOutFolder_clicked()
{

}

void MainWindow::on_doubleSpinBoxFixMinDisp_valueChanged(double arg1)
{
    //ProcessImages();
    ShowImages();
}

void MainWindow::on_doubleSpinBoxFixMaxDisp_valueChanged(double arg1)
{
    //ProcessImages();
    ShowImages();
}

void MainWindow::on_checkBoxShowGray_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_checkBoxShowGradient_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxFixMinDispGrad_valueChanged(double arg1)
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxFixMaxDispGrad_valueChanged(double arg1)
{
     ShowImages();
}

void MainWindow::on_spinBoxGradientSchape_valueChanged(int arg1)
{
    ProcessImages();
}

void MainWindow::on_spinBoxGradThreshold_valueChanged(int arg1)
{
    ProcessImages();
}

void MainWindow::on_comboBoxGradient_currentIndexChanged(int index)
{
    ProcessImages();
}

void MainWindow::on_checkBoxShowMask_toggled(bool checked)
{
     ShowImages();
}

void MainWindow::on_comboBoxOutputMode_currentIndexChanged(int index)
{
     ProcessImages();
}

void MainWindow::on_checkBoxShowPC_toggled(bool checked)
{
     ShowImages();
}

void MainWindow::on_spinBoxTileToProcess_valueChanged(int arg1)
{
    ProcessTile();
}

void MainWindow::on_spinBoxHistScaleHeight_valueChanged(int arg1)
{
    ProcessTile();
}

void MainWindow::on_spinBoxHistScaleCoef_valueChanged(int arg1)
{
    ProcessTile();
}

void MainWindow::on_spinBoxHistBarWidth_valueChanged(int arg1)
{
    ProcessTile();
}

void MainWindow::on_checkBoxShowHist_toggled(bool checked)
{
    ProcessTile();
}

void MainWindow::on_checkBoxSaveOutput_toggled(bool checked)
{

}

void MainWindow::on_comboBoxDisplayRange_currentIndexChanged(int index)
{

}

void MainWindow::on_checkBoxProcessGradient_toggled(bool checked)
{
    ProcessImages();
}

void MainWindow::on_checkBoxMaskBackGround_toggled(bool checked)
{
    ProcessImages();
}

void MainWindow::on_checkBoxProcessTile_toggled(bool checked)
{
   ProcessImages();
}

void MainWindow::on_spinBoxTileSize_valueChanged(int arg1)
{
    ProcessImages();
}

void MainWindow::on_checkBoxShowTile_toggled(bool checked)
{
    ProcessTile();
}

void MainWindow::on_doubleSpinBoxTileScale_valueChanged(double arg1)
{
    ProcessTile();
}

void MainWindow::on_checkBoxShowTileOnImage_toggled(bool checked)
{
    ProcessTile();
}

void MainWindow::on_checkBoxShowLesionMask_toggled(bool checked)
{
    ProcessTile();
}

void MainWindow::on_spinBoxTileSizeX_valueChanged(int arg1)
{
    ProcessImages();
}
